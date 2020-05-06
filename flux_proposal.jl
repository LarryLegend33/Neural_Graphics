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



# NOTE Y HAT IS THE RETURN VALUE OF THE MODEL ALWAYS
# Y IS THE GROUNDTRUTH. GROUNDTRUTH COMES OUT IN 4D. HAVE TO REDUCE
# IT TO 2D B/C THE MODELS OUTPUT IN 2D

# will bring in latent variables symbols and params
include("gen_pose_model.jl");



#loss(ŷ, y) = mse(ŷ, y[:,1,1,:]);
loss(ŷ, y) = logitcrossentropy(ŷ, y[:,1,1,:]);
accuracy_threshold(x) = 1.25 * (1.0 / size(x)[1])

#loss(y, ŷ) = mse(ŷ, y[:,1,1,:]);

xyz_init_lookup = readdlm("xyz_by_rotation.txt", ',')
lv_symbols = [lv[1] for lv in latent_variables]
joints = ["elbow_r",
          "elbow_l",
          "hip",
          "heel_r",
          "heel_l"]



# rotation is encoded as 1 in K in a 72 item digital output
# this directly corresponds to a row of initial xyz coords before the delta

function accuracy(y, ŷ, acc_array)
    thr = accuracy_threshold(ŷ)
    call = [prob > thr ? 1f0 : 0f0 for prob in softmax(ŷ)]
    # something to figure out later: why == is false, isapprox is true
    # if you print call and y, they will show the same values
    # yet report false if you use == 

    # true pos
    if sum(call) >= 1 && isapprox(y,call)
        acc_array[1] += 1
    # false pos
    elseif sum(call) >= 1 && !isapprox(y,call)
        acc_array[2] += 1
    # true neg
    elseif sum(call) == 0 && isapprox(y,call)
        acc_array[3] += 1
    # false neg
    elseif sum(call) == 0 && !isapprox(y,call)
        acc_array[4] += 1
    end
    return acc_array
end

function roundto(x::Float64, mod::Float64)
    rem = x % mod
    if rem < mod / 2
        return round(x-rem, digits=1)
    else
        return round(x+mod-rem, digits=1)
    end
end


# OK setup will be to first pass the whole image into the rotation network.
# Next, you pass each patch into the patch network. Generative function
# decides the staggering on the patch initiation, so that it will suggest
# different locations on each iteration and will narrow down the 
# right location based on mismatches.

function ray_cast_depthcoords(depthcoords::Tuple, resolution::Int)
    x, y, depth = depthcoords
    x /= resolution
    y /= resolution
    view_frame = [(-.5, -.5, 1.09375),
                  (-.5, .5, 1.09375),
                  (.5, .5, 1.09375)]
    frame = [(v / (v[3] / depth)) for v in view_frame]
    min_x, max_x = frame[2][1], frame[3][1]
    min_y, max_y = frame[1][2], frame[2][2]
    x_proportion = ((max_x - min_x) * x) + min_x
    y_proportion = ((max_y - min_y) * y) + min_y
    camera_matrix =  [(1.0000, 0.0000,  0.0000,  0.0000)
                      (0.0000, 0.5000, -0.8660, -8.5000)
                      (0.0000, 0.8660,  0.5000,  5.0000)
                      (0.0000, 0.0000,  0.0000,  1.0000)]
    xyz_location = camera_matrix * (x_proportion, y_proportion, -1*depth)
    return xyz_location
end

function neural_proposal(depth_stimulus, patch_dim::Int, rot_net::Chain,
                         depth_net::Chain, patch_net::Chain)
    start_x = {:start_x} ~ uniform(1, patch_dim)
    start_y = {:start_y} ~ uniform(1, patch_dim)
    resx, resy = size(depth_stimulus)
    detected_rotation = rot_net(depth_stimulus)
    rot_z_proposal = deg2rad(onecold(detected_rotation), 0:20:340))
    joint_locations = Dict()
    for y in start_y:patchdim:im_height-patchdim
        for x in start_x:patchdim:im_width-patchdim
            patch = image[y:y+patchdim-1, x:x+patchdim-1]
            p̂ = patch_net(patch)
            call = [prob > accuracy_threshold(p̂) ? 1f0 : 0f0 for prob in softmax(p̂)]
            if call != zeros(len(p))
                key = joints[findfirst(isequal(1.0), call)]
                # center of patch
                xyd = (x+patchdim/2, y+patchdim/2, onecold(depth_net(patch), 7:.2:13)
                joint_locations[key] = ray_cast_depthcoords(xyd, resx)
            end
        end
    end
    xyz_no_delta = init_xyz(rotation)
    
    # have to now go through joint_locations and find out the 
    
    

function extract_deltas(rotation::Int, detected_location::Tuple, metric::Symbol)
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
            joint_code = zeros(num_joints)
            for ind in 1:size(xydepth,1)
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

function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    return Chain(
            x -> reshape(x, imgsize..., :),
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end

function make_training_data(num_samples::Int)
    trace = Gen.simulate(body_pose_model, ());
    image_size = size(trace[:image])
    patch_dim = 30
    rot_res = 20.0
    depth_res = .2
    one_in_k_rot = convert(Int, 360 / rot_res)
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
        (patches, codes, depths) = patches_w_joint_gts(trace[:image],
                                                     trace[:groundtruths],
                                                     patch_dim)
        patches_all = vcat(patches_all, patches)
        codes_all = vcat(codes_all, codes)
        patches_for_depth = vcat(patches_for_depth, [pt for (pt,d) in zip(patches, depths) if !isnan(d)])
        depths_all = vcat(depths_all, [onehot(
            roundto(d, .2) , onehot_depth) for d in depths if !isnan(d)])
    end
    patch_trainingset = Array{Float32}(undef, patch_dim, patch_dim, 
                                       1, size(patches_all, 1))
    for (i, p) in enumerate(patches_all) patch_trainingset[:,:,:, i] = p end
    codes_trainingset = Array{Float32}(undef, size(codes_all[1], 1), 1,  
                                       1, size(patches_all, 1))
    for (i, c) in enumerate(codes_all) codes_trainingset[:,:,:, i] = c end
    depth_trainingset = Array{Float32}(undef, size(onehot_depth, 1), 1, 1, size(depths_all, 1))
    for (i, d) in enumerate(depths_all) depth_trainingset[:,:,:, i] = d end
    patches_d_trainingset = Array{Float32}(undef, patch_dim, patch_dim, 
                                           1, size(patches_for_depth, 1))
    for (i, p) in enumerate(patches_for_depth) patches_d_trainingset[:,:,:, i] = p end
    tset_dictionary = Dict("patches"=>patch_trainingset,
                           "codes_gt"=>codes_trainingset,
                           "patches_depth"=>patches_d_trainingset,
                           "depth_gt"=>depth_trainingset,
                           "rotation_gt"=>rotation_groundtruths,
                           "gen images"=>generated_images)
    return tset_dictionary
end



@with_kw mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 2       # batch size
    epochs = 200           # number of epochs
    training_samples = 100
    validation_samples = 20
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1 	 # report every `infotime` epochs
    save_every_n_epochs = epochs / 2   # Save the model every x epochs.
    tblogger = true       # log training with tensorboard
    savepath = "/Users/nightcrawler2/Neural_Graphics/logging"
end


# accuracy for me is simply the number of times in a batch the
# call is correct vs incorrect. 
function eval_validation_set(loader, model, device)
    total_loss = 0f0
    accuracy_array = zeros(4)
    false_neg = 0
    # switch accuracy threshold to 1/length of the input plus a
    # value -- can bias it up by a quarter of the value. 
    for (image, gt) in loader
        image, gt = image |> device, gt |> device
        ŷ = model(image)
        total_loss += loss(ŷ, gt)
        accuracy_array = accuracy(gt, ŷ, accuracy_array)
    end
    return (loss = round(total_loss, digits=4),
            true_pos = accuracy_array[1], 
            false_pos = accuracy_array[2],
            true_neg = accuracy_array[3],
            false_neg = accuracy_array[4])
end
    # figure this out first, then try to figure out how to
    # write a gen model to make a NN and mcmc the params
# for good performance.

function indep_validation(nn_model::Chain, validation_set, acc_thresh)
    validation_loader = DataLoader(
        validation_set...,
        batchsize=1)
    test_model_performance = eval_validation_set(
        validation_loader, 
        nn_model,
        cpu)

    return test_model_performance
end
    

function train_nn_on_dataset(nn_model::Chain, nn_args::Args,
                             validation_set, training_set)
    nn_args.seed > 0 && Random.seed!(nn_args.seed)
    use_cuda = nn_args.cuda && CUDAapi.has_cuda_gpu()
    # make thresh 25% higher than uniform. works perfect for
    # patches and depths
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end
    validation_loader = DataLoader(
        validation_set...,
        batchsize=1)
    training_loader = DataLoader(
        training_set...,
        batchsize=nn_args.batchsize, shuffle=true)
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
                    @info "train" loss=test_model_performance.loss
                    @info "train" true_pos=test_model_performance.true_pos
                    @info "train" false_pos=test_model_performance.false_pos
                    @info "train" true_neg=test_model_performance.true_neg
                    @info "train" false_neg=test_model_performance.false_neg
            end
            epoch == 0 && run(`tensorboard --logdir logging`, wait=false)
        end
        for (sample, groundtruth) in training_loader
            sample, groundtruth = sample |> device, groundtruth |> device
            grads = Flux.gradient(nn_params) do
                ŷ = nn_model(sample)
                loss(ŷ, groundtruth)
           #     loss(ŷ, groundtruth[:, :, 1, :])
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
    return nn_model
end    

nn_args = Args(epochs=100)
println(nn_args.batchsize)
#validation_data = make_training_data(nn_args.validation_samples)
#training_data = make_training_data(nn_args.training_samples)
rotation_net = LeNet5(;imgsize=(256,256,1), nclasses=length(0:20:355))

patch_net = LeNet5(;imgsize=(30,30,1), nclasses=5)
depth_net = LeNet5(;imgsize=(30,30,1), nclasses=length(7:.2:13))

# Absolutely perfect using logitcrossentropy as the
# loss function
# rotation_model = train_nn_on_dataset(rotation_net, 
#                                      nn_args,
#                                      (training_data["gen images"],
#                                       training_data["rotation_gt"]),
#                                      (training_data["gen images"],
#                                       training_data["rotation_gt"]))

#indep_validation(rotation_model, (training_data["gen images"], training_data["rotation_gt"]), .1)

#     (training_data["patches"], training_data["codes_gt"]))


# WORKS AWESOME WITH PROBABILITY THRESH SET AT .25
# really examine the loss function's inner workings.

# unreal performance using logitcrossentropy as the loss function
# in 10 epochs get 200 true positives. run a bit longer? 
patch_model = train_nn_on_dataset(
    patch_net,
    nn_args,
    (training_data["patches"], training_data["codes_gt"]),
    (training_data["patches"], training_data["codes_gt"]))

# completely perfect calls after 200 epochs with
# accuracy threshold set well (1.25 value above)
# depth_model = train_nn_on_dataset(
#     depth_net,
#     nn_args,
#     (training_data["patches_depth"], training_data["depth_gt"]),
#     (training_data["patches_depth"], training_data["depth_gt"]))



# OK so now custom proposal has to take the patch IDs and 
# convert them to deltas in the joint positions 


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

