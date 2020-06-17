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
using StatsBase
import BSON
import DrWatson: savename, struct2dict
import ProgressMeter

# array syntax: undef initializes the array with nans
# the rest are dimensions of the array.
# 28 x 28 x 1 x 10 means each element is a 28x28 image and there
# are 10 elements. 

# NOTE Y HAT IS THE RETURN VALUE OF THE MODEL ALWAYS
# Y IS THE GROUNDTRUTH. GROUNDTRUTH COMES OUT IN 4D. HAVE TO REDUCE
# IT TO 2D B/C THE MODELS OUTPUT IN 2D

include("gen_pose_model.jl");
#loss(ŷ, y) = mse(ŷ, y[:,1,1,:]);
loss(ŷ, y) = logitcrossentropy(ŷ, y[:,1,1,:]);
accuracy_threshold(x) = 1.25 * (1.0 / size(x)[1])
xyz_init_lookup = readdlm("xyz_by_rotation.txt", ',')
lv_symbols = [lv[1] for lv in latent_variables]
joints = [:elbow_r, 
          :elbow_l,
          :hip,
          :heel_r,
          :heel_l]

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
    groundtruths = Gen.get_retval(trace)
    patch_dim = 30
    rot_res = 5.0
    depth_res = .2
    one_in_k_rot = convert(Int, 360 / rot_res)
    # new priors yield wider depth values. 
    onehot_depth = 5:depth_res:15
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
                                                     groundtruths,
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


#  weight decay of .1 stops loss from decreasing. batchsize 2 is fine.
# batchsize 5 w/ no decay is fast and effective. by epoch 90 is 100% correct.
# so code works for multiple different batch sizes, for both rot and patch nets. 

@with_kw mutable struct Args
    η = 3e-4             # learning rate
    λ = 1e-4            # L2 regularizer param, implemented as weight decay
    batchsize = 5       # batch size
    epochs = 100           # number of epochs
    training_samples = 100
    validation_samples = 5
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1 	 # report every `infotime` epochs
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

function indep_validation(nn_model::Chain, validation_set)
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
                epoch == 0 && run(`tensorboard --logdir logging`, wait=false)
                set_step!(tblogger, epoch)
                with_logger(tblogger) do
                    @info "train" loss=test_model_performance.loss
                    @info "train" true_pos=test_model_performance.true_pos
                    @info "train" false_pos=test_model_performance.false_pos
                    @info "train" true_neg=test_model_performance.true_neg
                    @info "train" false_neg=test_model_performance.false_neg
                end
            end
        end
        for (samples, groundtruth) in training_loader
            samples, groundtruth = samples |> device, groundtruth |> device
            grads = Flux.gradient(nn_params) do
                ŷ = nn_model(samples)
                loss(ŷ, groundtruth)
            end
            Flux.Optimise.update!(opt, nn_params, grads)
            ProgressMeter.next!(p)   # comment out for no progress bar
        end
    end
    return nn_model
end


# going to do logging in here. the status bar is fine for the training networks
# just to make sure they are learning -- but you know this is true.
# run independent validation on the network after each learning epoch log this to TB. 


function train_on_model(iter::Int, rotation_net::Chain, depth_net::Chain,
                        patch_net::Chain, logger::TBLogger, nn_args::Args, finish::Int)

    iter == 1 && run(`tensorboard --logdir logging`, wait=false)

    if nn_args.cuda && CUDAapi.has_cuda_gpu()
        device = gpu
    else
        device = cpu
    end
    
    validation_data = make_training_data(nn_args.validation_samples)
    training_data = make_training_data(nn_args.training_samples)
    test_rot_performance = eval_validation_set(
        DataLoader(validation_data["gen images"],
                   validation_data["rotation_gt"], batchsize=1), 
        rotation_net, device)
    test_patch_performance = eval_validation_set(
        DataLoader(validation_data["patches"],
                   validation_data["codes_gt"], batchsize=1),
        patch_net, device)
    test_depth_performance = eval_validation_set(
        DataLoader(validation_data["patches_depth"],
                   validation_data["depth_gt"], batchsize=1),
        depth_net, device)
    
    set_step!(logger, iter)
    with_logger(logger) do
        @info "rotation" loss_r=test_rot_performance.loss
        @info "rotation" true_pos_r=test_rot_performance.true_pos
        @info "rotation" false_pos_r=test_rot_performance.false_pos
        @info "rotation" true_neg_r=test_rot_performance.true_neg
        @info "rotation" false_neg_r=test_rot_performance.false_neg
        @info "patches" loss_p=test_patch_performance.loss
        @info "patches" true_pos_p=test_patch_performance.true_pos
        @info "patches" false_pos_p=test_patch_performance.false_pos
        @info "patches" true_neg_p=test_patch_performance.true_neg
        @info "patches" false_neg_p=test_patch_performance.false_neg
        @info "depth" loss_d=test_depth_performance.loss
        @info "depth" true_pos_d=test_depth_performance.true_pos
        @info "depth" false_pos_d=test_depth_performance.false_pos
        @info "depth" true_neg_d=test_depth_performance.true_neg
        @info "depth" false_neg_d=test_depth_performance.false_neg
    end
    
    rotation_net = train_nn_on_dataset(rotation_net, nn_args,
                                       (training_data["gen images"], training_data["rotation_gt"]),
                                       (training_data["gen images"], training_data["rotation_gt"]))
    patch_net = train_nn_on_dataset(patch_net, nn_args,
                                    (training_data["patches"], training_data["codes_gt"]),
                                    (training_data["patches"], training_data["codes_gt"]))
    depth_net = train_nn_on_dataset(depth_net, nn_args,
                                    (training_data["patches_depth"], training_data["depth_gt"]),
                                    (training_data["patches_depth"], training_data["depth_gt"]))

    if iter % 2 == 0 || iter == finish
        !ispath(nn_args.savepath) && mkpath(nn_args.savepath)
        modelpath = joinpath(nn_args.savepath, "neural_proposal.bson") 
        let rnet=cpu(rotation_net), dnet=cpu(depth_net), pnet=cpu(patch_net)
            BSON.@save modelpath rnet dnet pnet nn_args
        end
    end
    if iter == finish
        return rotation_net, depth_net, patch_net, training_data, validation_data
    else
        train_on_model(iter+1, rotation_net, depth_net, patch_net, logger, nn_args, finish)
    end
end



# UNCOMMENT FOR TRAINING
#Note on round 1, batch size of 2, no softmax in net. Poor results. 
#Round 2, softmax is directly in the net. Are doing gradient on softmax results.
#Round 3, if it still doesn't look promising, use an optimizer. 

rotation_net = LeNet5(;imgsize=(256,256,1), nclasses=length(0:5:355))
patch_net = LeNet5(;imgsize=(30,30,1), nclasses=5)
depth_net = LeNet5(;imgsize=(30,30,1), nclasses=length(5:.2:15))
nn_args = Args(tblogger=false)
tblogger = TBLogger(nn_args.savepath, tb_overwrite)
set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
@info "TensorBoard logging at \"$(nn_args.savepath)\""
@info "Start Training"
rotation_net, depth_net, patch_net, training, validation  = train_on_model(
    1, rotation_net, depth_net, patch_net,
    tblogger, nn_args, 5)
trace = Gen.simulate(body_pose_model, ());
# proposed_deltas = neural_detection(trace[:image], 30, bs[:rnet],
#                                    bs[:dnet], bs[:pnet])
#trained_bson = BSON.load("./logging/neural_proposal.bson")


#indep_validation(rotation_model, (training_data["gen images"], training_data["rotation_gt"]), .1)

#     (training_data["patches"], training_data["codes_gt"]))


