using Flux
using Flux: crossentropy, logitcrossentropy, mse, onehot, onecold
using Gen
using Parameters: @with_kw
using Logging
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using CUDAapi
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay, ADAM
using DelimitedFiles
using Debugger
import BSON
import DrWatson: savename, struct2dict
import ProgressMeter


# Do three things here
# First, just create a model that can be trained on
# the generative model.

# array syntax: undef initializes the array with nans
# the rest are dimensions of the array.
# 28 x 28 x 1 x 10 means each element is a 28x28 image and there
# are 10 elements. lsp-mode



# will bring in latent variables symbols and params
include("gen_pose_model.jl");

loss(y, ŷ) = Flux.mse(ŷ, y);
xyz_init_lookup = readdlm("xyz_by_rotation.txt", ',')
lv_symbols = [lv[1] for lv in latent_variables]

# rotation is encoded as 1 in K in a 72 item digital output
# this directly corresponds to a row of initial xyz coords before the delta

function roundto(x::Float64, mod::Float64)
    rem = x % mod
    if rem < mod / 2
        return round(x-rem, digits=1)
    else
        return round(x+mod-rem, digits=1)
    end
end


function extract_deltas(rotation::Int, detected_locations::Dict)
    rotated_xyz = init_xyz(rotation)
    deltas = Dict([(lv, detected_locations[lv] - rotated_xyz[lv]) for lv in lv_symbols])
    # THIS WILL BE FED BACK AS THE CANDIDATE DELTAS
    return deltas
end

init_xyz(rotation) = Dict([(lv, xyz) for (lv, xyz) in  zip(
    lv_symbols[1:length(detected_locations)],
    xyz_init_lookup[rotation, :])])

# HAVE TO WRITE A DEPTH TO WORLD CALCULATOR THAT
# IS A HARD-CODED MATRIX TRANSFORM.

# FORM OF NN OUTPUT IS 

function world_groundtruth(trace)
    init_xyz_index = round(rad2deg(trace[:rot_z]) / 5)
    rotated_xyz = init_xyz(init_xyz_index)
    world_coords = Dict([
        (lv, trace[lv] + rotated_xyz[lv]) for lv in lv_symbols[1:end-3]])
    return world_coords
end


function patches_w_joint_gts(image::Array{Float64, 2},
                             xydepth::Array{Float64, 2}, patchdim::Int)
    im_width, im_height = size(image)
    num_joints = 5
    start_x = rand(1:patchdim)
    start_y = rand(1:patchdim)
    patches = []
    codes = []
    depths = []
    for y in start_y:patchdim:im_height-patchdim
        for x in start_x:patchdim:im_width-patchdim
            patch = image[y:y+patchdim-1, x:x+patchdim-1]
            println("PATCH SIZE")
            println(size(patch))
            joint_code = zeros(num_joints)
            for ind in 1:size(xydepth)[1]
                if (y < xydepth[
                    ind,2] < y+patchdim && x < xydepth[
                        ind,1] < x + patchdim)
                    joint_code[ind] = 1
                end
            end
            if sum(joint_code) == 1
                push!(depths, xydepth[findfirst(isequal(1), joint_code), 3])
            else
                push!(depths, NaN)
            end
            push!(patches, patch)
            push!(codes, joint_code)
        end
    end
    return patches, codes, depths
end


function make_training_data(num_samples::Int)
    trace = Gen.simulate(body_pose_model, ());
    image_size = size(trace[:image])
    patch_dim = 30
    rot_res = 5.0
    depth_res = .2
    one_in_k_rot = 72
    onehot_depth = 7:depth_res:13
    generated_images = Array{Float32}(undef, image_size...,
                                      1, num_samples)
    rotation_groundtruths = Array{Float32}(undef, one_in_k_rot,
                                           1, 1, num_samples)
    patches_all = []
    codes_all = []
    patches_for_depth = []
    depths_all = []
    for i in 1:num_samples
        trace = Gen.simulate(body_pose_model, ());
        generated_images[:,:,:,i] = trace[:image]
        rotation_groundtruths[:,:,:,i] = onehot(
            roundto(rad2deg(trace[:rot_z]), rot_res), 0:rot_res:360-rot_res)
        patches, codes, depths = patches_w_joint_gts(trace[:image],
                                                   trace[:groundtruths],
                                                   patch_dim)
        patches_all = vcat(patches_all, patches)
        codes_all = vcat(codes_all, codes)
        patches_for_depth = vcat(patches_for_depth, [pt for (pt,d) in zip(patches, depths) if !isnan(d)])
        depths_all = vcat(depths_all, [onehot(
            roundto(d, .2) , onehot_depth) for d in depths if !isnan(d)])
    end
    println("DEPTHS ALL")
    println(depths_all)
    patch_trainingset = Array{Float32}(undef, patch_dim, patch_dim, 
                                       1, size(patches_all)[1])
    for (i, p) in enumerate(patches_all) patch_trainingset[:,:,:, i] = p end
    codes_trainingset = Array{Float32}(undef, size(codes_all[1])[1], 1,  
                                       1, size(patches_all)[1])
    for (i, c) in enumerate(codes_all) codes_trainingset[:,:,:, i] = c end
    depth_trainingset = Array{Float32}(undef, size(onehot_depth)[1], 1, 1, size(depths_all)[1])
    for (i, d) in enumerate(depths_all) depth_trainingset[:,:,:, i] = d end
    patches_d_trainingset = Array{Float32}(undef, patch_dim, patch_dim, 
                                           1, size(patches_for_depth)[1])
    for (i, p) in enumerate(patches_for_depth) patches_d_trainingset[:,:,:, i] = p end             
    return patch_trainingset, codes_trainingset, depth_trainingset, patches_d_trainingset
end



@with_kw mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 10       # batch size
    epochs = 100           # number of epochs
    training_images = 10
    validation_images = 10
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1 	 # report every `infotime` epochs
    save_every_n_epochs = epochs / 5   # Save the model every x epochs.
    tblogger = true       # log training with tensorboard
    savepath = "/Users/nightcrawler2/Neural_Graphics/logging"
end



# LeNet like architecture to start
# think metaprogramming here. want to make a generator for
# conv layer structure. probably number of conv layers
function base_flux_model()
    image_size = (128, 128)
    kernel_size = (3, 3)
    edge_pad = (1, 1)
    # image isn't included in latent_variable descriptor
    output_size = length(latent_variables)
    nn_model = Chain(Conv(kernel_size, 1=>32, pad=edge_pad, relu),
                     MaxPool((2, 2)),
                     Conv(kernel_size, 32=>32, pad=edge_pad, relu),
                     MaxPool((2, 2)),
#                     Conv(kernel_size, 32=>32, pad=edge_pad, relu),
                     x -> reshape(x, :, size(x, 4)),
                     Dense(32^3, 128, relu), 
                     Dense(128, output_size, relu), 
                     softmax)
    return nn_model
end



function eval_validation_set(loader, model, device)
    total_loss = 0f0
    accuracy = 0
    # i would define accuracy as the number of
    # body points that are within .1
    for (image, gt) in loader
        image, gt = image |> device, gt |> device
        ŷ = model(image)
        total_loss += loss(ŷ, gt[1,:,1,1]) 
        accuracy += sum([abs(diff) < .1 ? 1 : 0 for diff in ŷ-gt[1,:,1,1]])
    end
    return (loss = round(total_loss, digits=4), acc = round(accuracy, digits=4))
end
    # figure this out first, then try to figure out how to
    # write a gen model to make a NN and mcmc the params
    # for good performance. 

function train_nn_on_dataset(nn_model::Chain; kws...)
    nn_args = Args(; kws...)
    nn_args.seed > 0 && Random.seed!(nn_args.seed)
    use_cuda = nn_args.cuda && CUDAapi.has_cuda_gpu()
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
    validation_loader = DataLoader(
        make_pose_data(nn_args.validation_images)...,
        batchsize=1)
    training_loader = DataLoader(
        make_pose_data(nn_args.training_images)...,
        batchsize=nn_args.batchsize)
    nn_params = params(nn_model)
    opt = Optimiser(ADAM(nn_args.η), WeightDecay(nn_args.λ))

    if nn_args.tblogger 
        tblogger = TBLogger(nn_args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(nn_args.savepath)\""
    end

    @info "Start Training"



    for epoch in 0:nn_args.epochs
        p = ProgressMeter.Progress(length(training_loader))
        if epoch % nn_args.infotime == 0
            test_model_performance = eval_validation_set(
                validation_loader, 
                nn_model,
                device)
            println("Epoch: $epoch Validation: $(test_model_performance)")            
            if nn_args.tblogger
                set_step!(tblogger, epoch)
                with_logger(tblogger) do
                    @info "train" loss=test_model_performance.loss acc=test_model_performance.acc
            end
            epoch == 0 && run(`tensorboard --logdir logging`, wait=false)
        end

        
        for (image, groundtruth) in training_loader
            image, groundtruth = image |> device, groundtruth |> device
            grads = Flux.gradient(nn_params) do
                ŷ = nn_model(image)
                loss(ŷ, groundtruth[1, :, 1, :])
            end
            Flux.Optimise.update!(opt, nn_params, grads)
            ProgressMeter.next!(p)   # comment out for no progress bar
        end
                
        if epoch > 0 && epoch % nn_args.save_every_n_epochs == 0
            !ispath(nn_args.savepath) && mkpath(nn_args.savepath)
            modelpath = joinpath(nn_args.savepath, "nn_model.bson") 
            let model=cpu(nn_model), nn_args=struct2dict(nn_args)
                BSON.@save modelpath nn_model epoch nn_args
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
    end
    return nn_model, training_loader, validation_loader
end    


#nn_mod = base_flux_model();
                      #trained_model, training_data, validation_data = train_nn_on_dataset(nn_mod)

make_training_data(5);
    
# function train_flux_with_model(num_batches::Int, batch_size::Int)

#     # nn_model will be a gen program in the future
    
#  end     
 
# may be useful to have network know the rotation before searching, and
# be trained with rotation as an input, coded as a 1 in K addon to the image. 
# ORDER OF NEURAL NETWORK HAS TO BE ROTATION FIRST. THEN RUN
# FILTERS OVER PATCHES OF INPUT. EACH PATCH WILL HAVE A DIGITAL OUTPUT
# THAT NOTES THE PRESENCE OR ABSENCE OF A FEATURE. CUSTOM PROPOSAL WILL
# CHANGE THE PHASE OF THE FILTERS SO THAT THE CENTER OF THE FILTER
# WILL BE IN DIFFERENT SPOTS, AND UNIQUE OUTPUTS WILL ARISE (i.e. the different
# phases will resolve XY more accurately). NEXT EACH PATCH IS PASSED INTO
# A DEPTH CALCULATOR. THIS IS A NET TRAINED ON PATCHES WITH KNOWN DEPTHS.
# OUTPUT IS A XY AND DEPTH FOR EACH JOINT. 

