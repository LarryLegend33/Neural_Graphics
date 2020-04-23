using Gen
using FileIO
using ImageIO
using ImageFiltering: imfilter, Kernel
using ColorTypes
using FixedPointNumbers
using Base.Iterators
using DelimitedFiles


latent_variables = [(:elbow_r_x, [-.2, .2]),
                    (:elbow_r_y, [0, .4]),
                    (:elbow_r_z, [0, .3]),
                    (:elbow_l_x, [-.2, .2]),
                    (:elbow_l_y, [-.2, .2]),
                    (:elbow_l_z, [0, .3]),
                    (:hip_z, [-.2, .2]),
                    (:heel_r_x, [-.2, .2]),
                    (:heel_r_y, [0, .2]), 
                    (:heel_r_z, [-.2, .2]),
                    (:heel_l_x, [-.2, .2]),
                    (:heel_l_y, [0, .2]), 
                    (:heel_l_z, [-.2, .2]),
                    (:elbow_r_rot, [0, π/3]),
                    (:elbow_l_rot, [0, π/3]),
                    (:rot_z, [0,π])]




struct NoisyMatrix <: Gen.Distribution{Matrix{Float64}} end

struct GaussianNoisyGroundtruths <: Gen.Distribution{Array{Float64, 2}} end

const noisy_matrix = NoisyMatrix()

const noisy_groundtruths = GaussianNoisyGroundtruths()

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


# You never call this; it is a standin -- but fix it eventually. 
# function Gen.logpdf(::GaussianNoisyGroundtruths, x::Array{Float64, 2}, mu::Array{U}, noise::T) where {U<:Real,T<:Real}
# end
function Gen.logpdf(::GaussianNoisyGroundtruths, x::Matrix{Float64}, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end



function Gen.random(::GaussianNoisyGroundtruths, mu::Array{U, 2}, noise::T) where {U<:Real,T<:Real}
    mat = copy(mu)
    (w, h) = size(mu)
    for i=1:w
        for j=1:h
            mat[i, j] = normal(mu[i, j], noise)
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


@gen function body_pose_model()
    # locations of relevant joints
    pose_params = [({lv} ~ uniform(win[1], win[2])) 
                   for (lv, win) in latent_variables]
    println(pose_params)
    depth_image, two_d_groundtruth = render_pose(pose_params, "depth")
    blurred_depth_image = imfilter(depth_image, Kernel.gaussian(1))
    noisy_image = ({ :image } ~ noisy_matrix(blurred_depth_image, 0.1))
    gaussian_groundtruths = ({ :groundtruths } ~ noisy_groundtruths(two_d_groundtruth, .001))
    return blurred_depth_image
end


function build_initial_positions()
    open("xyz_by_rotation.txt", "w") do file
        for rotation in 0:5:10
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




    
        # make a dictionary with :elbow_l_x, etc.
        
        

#build_initial_positions()
trace = Gen.simulate(body_pose_model, ());
            


# This is the real structure of the model -- i.e. where to look. 




#choice_list = [trace[lv] for lv in latent_variables];

# spawn two processes that render the depth and wire images based on gen model
# if you want, you can initialize with a shell script that puts
# virtual frame buffers on servers with xvfb. then you have to remove the
# bpy_depth_standalone main commands and expose the functions called in them
# i wonder though if this wouldn't work already. 


# neural_proposal.jl contains a gen function that
# proposess beta or normally distributed NN outputs.
# proposals in gen have to be gen functions with traced choices. 
