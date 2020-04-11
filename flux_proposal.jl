using Flux
using Flux: crossentropy, logitcrossentropy, mse
using Gen
using Parameters: @with_kw
using Logging
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
using CUDAapi
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay, ADAM
import BSON
import DrWatson: savename, struct2dict
import ProgressMeter

# Do three things here
# First, just create a model that can be trained on
# the generative model.

# array syntax: undef initializes the array with nans
# the rest are dimensions of the array.
# 28 x 28 x 1 x 10 means each element is a 28x28 image and there
# are 10 elements. 


include("gen_pose_model.jl");

loss(y, ŷ) = mse(ŷ, y); 

function make_pose_data(num_samples::Int)
    trace = Gen.simulate(body_pose_model, ());
    image_size = size(trace[:image])
    generated_images = Array{Float32}(undef, image_size...,
                                      1, num_samples)
    groundtruths = Array{Float32}(undef, 1,
                                  length(latent_variables),
                                  1, num_samples)
    for i in 1:num_samples
        trace = Gen.simulate(body_pose_model, ());
        generated_images[:,:,:,i] = trace[:image]
        groundtruths[:,:,:,i] = 
            [Float32(trace[lv]) for lv in latent_variables]
    end
    labeled_data = (generated_images, groundtruths)
    return labeled_data
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


nn_mod = base_flux_model();
trained_model, training_data, validation_data = train_nn_on_dataset(nn_mod) 

    
# function train_flux_with_model(num_batches::Int, batch_size::Int)

#     # nn_model will be a gen program in the future
    
#  end     
 



