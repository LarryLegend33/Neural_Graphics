using Gen
using FileIO
using ImageIO
using ImageFiltering: imfilter, Kernel
using ColorTypes
using FixedPointNumbers
using Base.Iterators
using DelimitedFiles
using Flux
using Flux: crossentropy, logitcrossentropy, mse, onehot, onecold
using Luxor;
import BSON


# currently each of these windows is loosely
# established so that it doesn't pull the joint off of the model
# which caused huge headaches. next iteration should
# establish a standard pose followed by bending of joint angles.
# positioning joints carries huge technical baggage.
# also next iteration must assure non-collision of body parts.
# another hurdle is getting rid of blender. each rendering
# takes seconds and prints ~1000 lines of code to the screen.

trained_nn_bson = BSON.load("./logging/neural_proposal.bson")
xyz_init_lookup = readdlm("xyz_by_rotation.txt", ',')
joints = [:elbow_r, 
          :elbow_l,
          :hip,
          :heel_r,
          :heel_l]

latent_variables = [(:elbow_r_x, [-.5, .5]),
                    (:elbow_r_y, [-1.2, 1.2]),
                    (:elbow_r_z, [0, 2.9]),
                    (:elbow_l_x, [-.5, .5]),
                    (:elbow_l_y, [-1.2, 1.2]),
                    (:elbow_l_z, [0, 2.9]),
                    (:hip_z, [0, 1]),
                    (:heel_r_x, [-2.4, 2.4]),
                    (:heel_r_y, [-2, 2]), 
                    (:heel_r_z, [0, 2]),
                    (:heel_l_x, [-2.4, 2.4]),
                    (:heel_l_y, [-2, 2]), 
                    (:heel_l_z, [0, 2]),
                    (:elbow_r_rot, [-π, 0]),
                    (:elbow_l_rot, [-π, 0]),
                    (:rot_z, [0,(2*π) - .087])]

init_xyz(rotation) = Dict([(lv, xyz) for (lv, xyz) in  zip(
    [l[1] for l in latent_variables[1:13]],
    [xyz_init_lookup[rotation, 1:6];
     xyz_init_lookup[rotation, 9:end]])])

@gen function neural_proposal(current_trace, deltas)
    pose_params = [({lvk} ~ normal(deltas[lvk], 1)) for lvk in keys(deltas) if lvk == :rot_z]
end;
    
function neural_mh_update(tr, proposed_values)
    (tr, _) = mh(tr, neural_proposal, (proposed_values,))
    tr
end

function neural_inference(constraints, proposed_values)
    (tr, )  = generate(body_pose_model, (), constraints)
    for iter = 1:20
        tr = neural_mh_update(tr, proposed_values)
    end
    tr
end


struct NoisyMatrix <: Gen.Distribution{Matrix{Float64}} end

const noisy_matrix = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Matrix{Float64}, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function Gen.random(::NoisyMatrix, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    mat = copy(mu)
    (w, h) = size(mu)
    for i=1:w
        for j=1:h
            mat[i, j] = mu[i, j] + randn() * noise
            if mat[i, j] > 1
                mat[i, j] = 1
            elseif mat[i, j] < 0
                mat[i, j] = 0
            end
        end
    end
    return mat
end


function render_pose(pose, render_type)
    run(`/Applications/Blender.app/Contents/MacOS/Blender -b HumanKTH.decimated.blend -P bpy_depth_standalone.py -- $pose $render_type`)
    image_png = FileIO.load("$render_type.png")
    groundtruths = readdlm("$render_type.txt", ',')
    image_matrix = convert(Matrix{Float32}, image_png)
    return image_matrix, groundtruths
end


function compare_renderings(trace_list)
    labels = ["observed.png", "mh_only.png", "nn_mh.png"]
    for (trace, label) in zip(trace_list, labels)
        pose = [trace[lv] for (lv, win) in latent_variables]
        render_pose(pose, "wire")
        mv("wire.png", label, force=true)
        if label == "observed.png"
            render_pose(pose, "depth")
            mv("depth.png", "observed_depth.png", force=true)
        end
    end
end     


@gen function body_pose_model()
    # locations of relevant joints
    pose_params = [({lv} ~ uniform(win[1], win[2])) 
                   for (lv, win) in latent_variables]
    depth_image, two_d_groundtruth = render_pose(pose_params, "depth")
    blurred_depth_image = imfilter(depth_image, Kernel.gaussian(1))
    noisy_image = ({ :image } ~ noisy_matrix(blurred_depth_image, 0.1))
    return two_d_groundtruth
end


function build_initial_positions()
    open("xyz_by_rotation.txt", "w") do file
        for rotation in 0:5:360
            pose = zeros(length(latent_variables))
            pose[length(latent_variables)] = deg2rad(rotation)
            render_pose(pose, "depth")
            joint_xyz = readdlm("bone_positions3D.txt", ',')
            line = reshape(
                transpose(joint_xyz), 1, size(joint_xyz)[1] * size(joint_xyz)[2])
#            line = string(line)[2:end-1]
            writedlm(file, line, ',')
        end
    end
    return
end


function ray_cast_depthcoords(depthcoords::Tuple, resolution::Int)
    x, y, depth = depthcoords
    x /= resolution
    y /= resolution
    view_frame = [[-.5, -.5, 1.09375],
                  [-.5, .5, 1.09375],
                  [.5, .5, 1.09375]]
    frame = [(v / (v[3] / depth)) for v in view_frame]
    min_x, max_x = frame[2][1], frame[3][1]
    min_y, max_y = frame[1][2], frame[2][2]
    x_proportion = ((max_x - min_x) * x) + min_x
    y_proportion = ((max_y - min_y) * y) + min_y
    camera_matrix = [1.0 0.0 0.0 0.0;
                     0.0 0.5 -0.866 -8.5;
                     0.0 0.866 0.5 5.0;
                     0.0 0.0 0.0 1.0]
    # need the 1 b/c blender works in 4D vectors for translation
    xyz_location = camera_matrix * [x_proportion, y_proportion, -1*depth, 1]
    return xyz_location
end


function neural_detection(depth_stimulus, patchdim::Int, rot_net::Chain,
                         depth_net::Chain, patch_net::Chain)
    start_x = 40
    start_y = 40
    resx, resy = size(depth_stimulus)
    r̂ = softmax(rot_net(depth_stimulus))
    r̂_weights = ProbabilityWeights(r̂[:,1])
    rotation_call = sample(r̂_weights)
    rot_z_proposal = deg2rad(rotation_call * 5)
    joint_xyd = Dict()
    joint_locations = Dict()
    max_probabilities = zeros(length(joints))
    for y in start_y:patchdim:resy-patchdim-40
        for x in start_x:patchdim:resx-patchdim-40
            patch = depth_stimulus[y:y+patchdim-1, x:x+patchdim-1]
            p̂ = softmax(patch_net(patch))
            max_prob, max_index = findmax(p̂)
            # can also sample here instead of finding max.
#            p̂_weights = ProbabilityWeights(p̂[:,1])
#            patch_call = sample(p̂_weights)
            if max_prob > max_probabilities[max_index]
                max_probabilities[max_index] = max_prob
                detected_depth = onecold(depth_net(patch), 5:.2:15)[1]
                xyd = (x+patchdim/2, y+patchdim/2, detected_depth)
                proj_3D = ray_cast_depthcoords(xyd, resx)
                key_joint = joints[max_index]
                if key_joint != :hip
                    joint_locations[Symbol(key_joint, "_x")] = proj_3D[1]
                    joint_locations[Symbol(key_joint, "_y")] = proj_3D[2]
                end
                joint_locations[Symbol(key_joint, "_z")] = proj_3D[3]
                joint_xyd[key_joint] = xyd
            end
        end
    end
    # each init coord is indexed by a symbol from LVs.
    xyz_baseline = init_xyz(rotation_call)
    # proposed deltas for all detected joints
    deltas = Dict(
        [(k, joint_locations[k] - xyz_baseline[k]) for k in keys(joint_locations)])
    deltas[:rot_z] = rot_z_proposal
    joint_xyd[:rot_z] = rot_z_proposal
    return deltas, joint_xyd
end


function draw_joints(joint_xyd)
    # multiply by 2 b/c wires are twice as high resolution
    image = readpng("observed.png")
    w = image.width
    h = image.height
    fname = "annoted_image.png"
    Drawing(w, h, fname)
    placeimage(image, 0, 0)
    setline(0.3)
    sethue("blue")
    fontsize(10)
    for jt in keys(joint_xyd)
        if jt != :rot_z
            xyd = joint_xyd[jt]
            label(string(jt), :NE, Point(xyd[1] * 2, xyd[2] * 2), leader=true, offset=25)
        else
            text(string("rotation = ", joint_xyd[:rot_z]), Point(10, 10))
        end
    end
    finish()
end
        

#build_initial_positions()

# RUN INFERENCE HERE. FIRST GENERATE AN IMAGE. MAKE CONSTRAINTS WHICH IS THE
# IMAGE CREATED BY THE TRACE. 

observed_trace = Gen.simulate(body_pose_model, ());
constraints = Gen.choicemap()
constraints[:image] = observed_trace[:image]
proposed_deltas, joint_xyd = neural_detection(observed_trace[:image],
                                              30,
                                              trained_nn_bson[:rnet],
                                              trained_nn_bson[:dnet],
                                              trained_nn_bson[:pnet])
no_nn_trace = neural_inference(constraints, Dict())
nn_trace = neural_inference(constraints, proposed_deltas)
compare_renderings([observed_trace, no_nn_trace, nn_trace])
draw_joints(joint_xyd)




# run these commands in the repl for now. for each iteration, save the image. 




    






            

#choice_list = [trace[lv] for lv in latent_variables];

# spawn two processes that render the depth and wire images based on gen model
# if you want, you can initialize with a shell script that puts
# virtual frame buffers on servers with xvfb. then you have to remove the
# bpy_depth_standalone main commands and expose the functions called in them
# i wonder though if this wouldn't work already. 


# neural_proposal.jl contains a gen function that
# proposess beta or normally distributed NN outputs.
# proposals in gen have to be gen functions with traced choices. 
