using Gen
using FileIO
using ImageIO
using ImageFiltering: imfilter, Kernel
using ColorTypes
using FixedPointNumbers
using Base.Iterators

latent_variables = [:rot_z, 
                    :elbow_r_loc_x,
                    :elbow_r_loc_y,
                    :elbow_r_loc_z,
                    :elbow_l_loc_x,
                    :elbow_l_loc_y,
                    :elbow_l_loc_z,
                    :elbow_r_rot,
                    :elbow_l_rot,
                    :hip_loc_z,
                    :heel_r_loc_x,
                    :heel_r_loc_y,
                    :heel_r_loc_z,
                    :heel_l_loc_x,
                    :heel_l_loc_y,
                    :heel_l_loc_z]

# I'm guessing this makes noisymatrix a subclass 
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
    image_matrix = convert(Matrix{Float32}, image_png)
    return image_matrix
end



@gen function body_pose_model()
    # locations of relevant joints
    pose_params = [({lv} ~ uniform(0, 1)) for lv in latent_variables]
    depth_image = render_pose(pose_params, "depth")
    blurred_depth_image = imfilter(depth_image, Kernel.gaussian(1))
    noisy_image = ({ :image } ~ noisy_matrix(blurred_depth_image, 0.1))
    return blurred_depth_image
end

trace = Gen.simulate(body_pose_model, ());


#choice_list = [trace[lv] for lv in latent_variables];

# spawn two processes that render the depth and wire images based on gen model
# if you want, you can initialize with a shell script that puts
# virtual frame buffers on servers with xvfb. then you have to remove the
# bpy_depth_standalone main commands and expose the functions called in them
# i wonder though if this wouldn't work already. 


# neural_proposal.jl contains a gen function that
# proposess beta or normally distributed NN outputs.
# proposals in gen have to be gen functions with traced choices. 
