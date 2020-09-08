using Makie
using AbstractPlotting.MakieLayout
using Gen
using LinearAlgebra
using LightGraphs
using MetaGraphs
using Random


# @dist function truncated_normal(mu, sig, lb, ub)
#     sample = normal(mu, sig)
#     if sample < lb || sample > ub
#         sample = truncated_normal(mu, sig, lb, ub)
#     end
# end    
    
#- One main question is whether we are going to try to reconstruct the identity after the fact. I.e. Are the xs and ys completely known in time and space? We can do simultaneous inference on x and y values wrt t. 




#- starting with init positions b/c this is the type of custom proposal you will get from the tectum. you won't get offests for free. this model accounts for distance effects and velocity effects by traversing the tree. but it's lousy b/c it is static at 3 dots and merge / split opportunities are slim.

#-PASS THIS FUNCTION A SET OF TIMES AND AN EMPTY METADIGRAPH TO START - MetaDiGraph()
#- Involution will be you remove the edge and subtract the velocity, sample a uniform XY.
#- New edge will be you add the velocity and sample from a gaussian. 

@gen function add_node(motion_tree::MetaDiGraph{Int64, Float64}, 
                       candidate_nodes::Array{Int64, 1})
    if isempty(candidate_nodes)
        add_vertex!(motion_tree)
        return motion_tree
    end
    dot = first(candidate_nodes)
    if isempty(inneighbors(motion_tree, dot))
        parent_to_node = { :parent } ~ bernoulli(.5)
    else
        parent_to_node = { :parent } ~ bernoulli(.2)
    end

    if parent_to_node
        add_vertex!(motion_tree)
        add_edge!(motion_tree, dot, nv(motion_tree))
        return motion_tree
    else
        add_node(motion_tree, filter(λ -> λ != dot, candidate_nodes))
    end
end    


@gen function assign_positions_and_velocities(motion_tree::MetaDiGraph{Int64, Float64}
                                              dot::Int64)
    position_variance = .2
    if dot > nv(motion_tree)
        return motion_tree
    else
        parent = inneighbors(motion_tree, dot)
        if isempty(parent)
            xinit = {(:xinit, dot)} ~ uniform(0, 1)
            yinit = {(:yinit, dot)} ~ uniform(0, 1)
            cov_func_x = {(:cov_tree_x, dot)} ~ covariance_prior()
            cov_func_y = {(:cov_tree_y, dot)} ~ covariance_prior()
        else
            parent_position = props(motion_tree, parent[1])[:Position]
            parent_velocity = props(motion_tree, parent[1])[:Velocity]
            xinit = {(:xinit, dot)} ~ truncated_normal(parent_position[1], position_variance, 0, 1)
            yinit = {(:yinit, dot)} ~ truncated_normal(parent_position[2], position_variance, 0, 1)                     
            cov_func_x = {(:cov_tree_x, dot)} ~ covariance_prior()
            cov_func_y = {(:cov_tree_y, dot)} ~ covariance_prior()
        end
        noise = { :noise } ~  gamma_bounded_below(1, 1, 0.01)
        covmat_x = compute_cov_matrix_vectorized(cov_func_x, noise, ts)
        covmat_y = compute_cov_matrix_vectorized(cov_func_y, noise, ts)
    # Sample from the GP using a multivariate normal distribution with
    # the kernel-derived covariance matrix.
        x_vel = {:x_vel} ~ mvnormal(zeros(length(ts)), cov_mat_x)
        y_vel = {:y_vel} ~ mvnormal(zeros(length(ts)), cov_mat_y)
# next think about exactly what format you want these in           
            
        

@gen function gen_dotmotion_recur(ts::Vector{Float64},
                                  motion_tree::MetaGraph{Int64, Float64},
                                  num_dots::Int64)
    if nv(motion_tree) < num_dots
        motion_tree_updated = add_node(motion_tree, shuffle(vertices(motion_tree)))
        gen_dotmotion_recur(ts, motion_tree_updated, num_dots)
    else
        motion_tree_assigned = assign_positions_and_velocities(motion_tree, 1)
        
    
        


        
    

# GREAT! MetaDiGraph is what you want.     

# @gen function generate_dotmotion(ts:Vector{Float64})
#     connected_var = .2
#     motion_tree = MetaGraph(SimpleDiGraph(3))
#     init_dot1 = [{ :xinit1 } ~ uniform(0,1), { :yinit1 } ~ uniform(0,1)]
#     tree_type = { :tree_type } ~ choose_tree_type()
#     println(tree_type)
#     if tree_type in ["OneFree", "Connected" , "SharedParent"]
#         add_edge!(motion_tree.graph, 1, 2)
#         init_dot2 = [{ :xinit2 } ~ normal(init_dot1[1], .1),
#                      { :yinit2 } ~ normal(init_dot1[2], .1)]
#     else
#         init_dot2 = [{ :xinit2 } ~ uniform(0,1), { :yinit2 } ~ uniform(0,1)]
#     end
#     if tree_type == "Connected"
#         add_edge!(motion_tree.graph, 2, 3)
#         init_dot3 = [{ :xinit3 } ~ normal(init_dot2[1], .1),
#                      { :yinit3 } ~ normal(init_dot2[2], .1)]
#     end 
#     if tree_type == "SharedParent"
#         add_edge!(motion_tree.graph, 1, 3)
#         init_dot3 = [{ :xinit3 } ~ normal(init_dot1[1], .1),
#                      { :yinit3 } ~ normal(init_dot1[2], .1)]
#     end
#     if tree_type in ["OneFree", "AllFree"]
#         init_dot3 = [{ :xinit3 } ~ uniform(0,1), { :yinit3 } ~ uniform(0,1)]
#     end
#     covariance_fn_x = [{(:cov_tree_x, i)} ~ covariance_prior() for i in range(1, stop=3)]
#     covariance_fn_y = [{(:cov_tree_y, i)} ~ covariance_prior() for i in range(1, stop=3)]
#     noise = { :noise } ~  gamma_bounded_below(1, 1, 0.01)
#     covmats_x = [compute_cov_matrix_vectorized(
#         covariance_fn_x[i], noise, ts) for i in range(1, stop=3)]
#     covmats_y = [compute_cov_matrix_vectorized(
#         covariance_fn_y[i], noise, ts) for i in range(1, stop=3)]

#     # Sample from the GP using a multivariate normal distribution with
#     # the kernel-derived covariance matrix.
#     xs = 
#     ys = [{(:ys, i)} ~ mvnormal(zeros(length(ts)), cov_matrix_list[i]) for i in range(1, stop=3)]
#     set_props!(motion_tree, 1, Dict(:Position=> init_dot1, :Velocity => ys[1]))
#     set_props!(motion_tree, 2, Dict(:Position=> init_dot2, :Velocity => ys[2]))
#     set_props!(motion_tree, 3, Dict(:Position=> init_dot3, :Velocity => ys[3]))
#     return motion_tree
# end

    
                                      



"""Create Makie Rendering Environment"""

function tree_to_coords(tree::MetaGraph{Int64, Float64}, num_dots::Int64)
    dotmotion = zeros(Float64, num_dots, size(props(tree, 1)[:Velocity])[1])
    # Assign first dot positions based on its initial XY position and velocities
    for i in range(1, stop=num_dots)
        dotmotion[i, :] = props(tree, i)[:Velocity] .+ props(tree, i)[:Position]
    end
    # should be a nicer way to map over this. for now its fine but clean up later.
    # but you want to build infrastructure to go up on dots. 
    if has_edge(tree, 1, 2)
        dotmotion[2, :] += props(tree, 1)[:Velocity]
    end
    if has_edge(tree, 1, 3)
        dotmotion[3, :] += props(tree, 1)[:Velocity]
    end
    if has_edge(tree, 2, 3)
        dotmotion[3, :] += props(tree, 2)[:Velocity]
    end
    dotmotion_tuples = [(dotmotion[1, i], dotmotion[2, i], dotmotion[3, i]) for i in range(
        1, size(dotmotion)[2])]
    return dotmotion_tuples
end


function render_simulation(num_iterations::Int64)
    res = 1000
    outer_padding = 0
    num_updates = 180
    n_rows = 1
    n_cols = 2
    scene, layout = layoutscene(outer_padding,
                            resolution = (2*res, res), 
                                backgroundcolor=RGBf0(0, 0, 0))
    ts = zeros(num_updates)
    trace = simulate(generate_dotmotion, (ts, ))
    motion_tree = get_retval(trace)
    return motion_tree
end
        # REST OF FUNCTION IS HERE -- CUT OFF AT "END"
        
#     dotmotion = tree_to_coords(motion_tree, 3)
#     axes = [LAxis(scene, backgroundcolor=RGBf0(0, 0, 0)) for i in 1:nrows, j in 1:ncols]
#     layout[1:nrows, 1:ncols] = axes
#     time_node = Node(1);
#     f(t, coords) = coords[t]

#     scatter!(axes[1], lift(t -> f(t, dotmotion), time_node), markersize=20px, color=RGBf0(255, 255, 255))
#     limits!(axes[1], BBox(0, 1, 0, 1))
#     scatter!(axes[2], lift(t -> f(t, dotmotion), time_node), markersize=20px, color=RGBf0(255, 255, 255))
#     limits!(axes[2], BBox(0, 1, 0, 1))
#     display(scene)

#     for i in range(1, stop=num_updates)
#         time_node[] = i
#         sleep(1/60)
#     end
# end
    
# Currently in makie_test. Takes a tree and renders the tree and the stimulus.


#- BELOW IS CODE FOR GENERATING TIME SERIES VIA GPs FROM 6.885 PSETS. IT'S ACTUALLY A GREAT STARTING POINT FOR GENERATING SYMBOLIC MOTION PATTERNS. BUT SIMPLIFY FOR NOW. GET RID OF COMPOSITE NODES AND SQUARED EXPONENTIAL MOTION. FOR NOW, JUST KEEP CONSTANT, LINEAR, AND PERIODIC.-#


    



"""Node in a tree where the entire tree represents a covariance function"""
abstract type Kernel end
abstract type PrimitiveKernel <: Kernel end
abstract type CompositeKernel <: Kernel end

"""Number of nodes in the tree describing this kernel."""
Base.size(::PrimitiveKernel) = 1
Base.size(node::CompositeKernel) = node.size


#- HERE EACH KERNEL TYPE FOR GENERATING TIME SERIES IS DEFINED USING MULTIPLE DISPATCH ON eval_cov AND eval_cov_mat. 

"""Constant kernel"""
struct Constant <: PrimitiveKernel
    param::Float64
end

eval_cov(node::Constant, x1, x2) = node.param

function eval_cov_mat(node::Constant, xs::Vector{Float64})
    n = length(xs)
    fill(node.param, (n, n))
end

"""Linear kernel"""
struct Linear <: PrimitiveKernel
    param::Float64
end

eval_cov(node::Linear, x1, x2) = (x1 - node.param) * (x2 - node.param)

function eval_cov_mat(node::Linear, xs::Vector{Float64})
    xs_minus_param = xs .- node.param
    xs_minus_param * xs_minus_param'
end

"""Squared exponential kernel"""
struct SquaredExponential <: PrimitiveKernel
    length_scale::Float64
end

eval_cov(node::SquaredExponential, x1, x2) =
    exp(-0.5 * (x1 - x2) * (x1 - x2) / node.length_scale)

function eval_cov_mat(node::SquaredExponential, xs::Vector{Float64})
    diff = xs .- xs'
    exp.(-0.5 .* diff .* diff ./ node.length_scale)
end

"""Periodic kernel"""
struct Periodic <: PrimitiveKernel
    scale::Float64
    period::Float64
end

function eval_cov(node::Periodic, x1, x2)
    freq = 2 * pi / node.period
    exp((-1/node.scale) * (sin(freq * abs(x1 - x2)))^2)
end

function eval_cov_mat(node::Periodic, xs::Vector{Float64})
    freq = 2 * pi / node.period
    abs_diff = abs.(xs .- xs')
    exp.((-1/node.scale) .* (sin.(freq .* abs_diff)).^2)
end

#-THESE NODES CREATE BIFURCATIONS IN THE TREE THAT GENERATE TWO NEW NODE TYPES, WHICH CAN MAKE THE FUNCTION A COMPOSITE OF MULTIPLE NODE INSTANCES AND TYPES-#

"""Plus node"""
struct Plus <: CompositeKernel
    left::Kernel
    right::Kernel
    size::Int
end

Plus(left, right) = Plus(left, right, size(left) + size(right) + 1)

function eval_cov(node::Plus, x1, x2)
    eval_cov(node.left, x1, x2) + eval_cov(node.right, x1, x2)
end

function eval_cov_mat(node::Plus, xs::Vector{Float64})
    eval_cov_mat(node.left, xs) .+ eval_cov_mat(node.right, xs)
end


"""Times node"""
struct Times <: CompositeKernel
    left::Kernel
    right::Kernel
    size::Int
end

Times(left, right) = Times(left, right, size(left) + size(right) + 1)

function eval_cov(node::Times, x1, x2)
    eval_cov(node.left, x1, x2) * eval_cov(node.right, x1, x2)
end

function eval_cov_mat(node::Times, xs::Vector{Float64})
    eval_cov_mat(node.left, xs) .* eval_cov_mat(node.right, xs)
end


#-THE COVARIANCE MATRIX WILL HAVE THE DIMENSIONS OF YOUR TIME SERIES IN X, AND DEFINES THE RELATIONSHIPS BETWEEN EACH TIMEPOINT. 

"""Compute covariance matrix by evaluating function on each pair of inputs."""
function compute_cov_matrix(covariance_fn::Kernel, noise, xs)
    n = length(xs)
    cov_matrix = Matrix{Float64}(undef, n, n)
    for i=1:n
        for j=1:n
            cov_matrix[i, j] = eval_cov(covariance_fn, xs[i], xs[j])
        end
        cov_matrix[i, i] += noise
    end
    return cov_matrix
end


"""Compute covariance function by recursively computing covariance matrices."""
function compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    n = length(xs)
    eval_cov_mat(covariance_fn, xs) + Matrix(noise * LinearAlgebra.I, n, n)
end

"""
Computes the conditional mean and covariance of a Gaussian process with prior mean zero
and prior covariance function `covariance_fn`, conditioned on noisy observations
`Normal(f(xs), noise * I) = ys`, evaluated at the points `new_xs`.
"""
function compute_predictive(covariance_fn::Kernel, noise::Float64,
                            xs::Vector{Float64}, ys::Vector{Float64},
                            new_xs::Vector{Float64})
    n_prev = length(xs)
    n_new = length(new_xs)
    means = zeros(n_prev + n_new)
#    cov_matrix = compute_cov_matrix(covariance_fn, noise, vcat(xs, new_xs))
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, vcat(xs, new_xs))
    cov_matrix_11 = cov_matrix[1:n_prev, 1:n_prev]
    cov_matrix_22 = cov_matrix[n_prev+1:n_prev+n_new, n_prev+1:n_prev+n_new]
    cov_matrix_12 = cov_matrix[1:n_prev, n_prev+1:n_prev+n_new]
    cov_matrix_21 = cov_matrix[n_prev+1:n_prev+n_new, 1:n_prev]
    @assert cov_matrix_12 == cov_matrix_21'
    mu1 = means[1:n_prev]
    mu2 = means[n_prev+1:n_prev+n_new]
    conditional_mu = mu2 + cov_matrix_21 * (cov_matrix_11 \ (ys - mu1))
    conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 * (cov_matrix_11 \ cov_matrix_12)
    conditional_cov_matrix = 0.5 * conditional_cov_matrix + 0.5 * conditional_cov_matrix'
    (conditional_mu, conditional_cov_matrix)
end

"""
Predict output values for some new input values
"""
function predict_ys(covariance_fn::Kernel, noise::Float64,
                    xs::Vector{Float64}, ys::Vector{Float64},
                    new_xs::Vector{Float64})
    (conditional_mu, conditional_cov_matrix) = compute_predictive(
        covariance_fn, noise, xs, ys, new_xs)
    mvnormal(conditional_mu, conditional_cov_matrix)
end

# kernel_types = [Constant, Linear, SquaredExponential, Periodic, Plus, Times]
# @dist choose_kernel_type() = kernel_types[categorical([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])];

kernel_types = [Constant, Linear, Periodic]
@dist choose_kernel_type() = kernel_types[categorical([0.5, 0.25, 0.25])]


# Prior on kernels
@gen function covariance_prior()
    # Choose a type of kernel
    kernel_type = { :kenrel_type } ~ choose_kernel_type()
    # If this is a composite node, recursively generate subtrees. For now, too complex. 
    if in(kernel_type, [Plus, Times])
        return kernel_type({ :left } ~ covariance_prior(), { :right } ~ covariance_prior())
    end
    # Otherwise, generate parameters for the primitive kernel.
    kernel_args = (kernel_type == Periodic) ? [{:scale} ~ uniform(0, 1), {:period} ~ uniform(0, 1)] : [{:param} ~ uniform(0, 1)]
    return kernel_type(kernel_args...)
end


@dist gamma_bounded_below(shape, scale, bound) = gamma(shape, scale) + bound


                                          
                                          
                                          
# Full model
@gen function cv_generation_model(xs::Vector{Float64})
    
    # Generate a covariance kernel
    covariance_fn = { :tree } ~ covariance_prior()
    
    # Sample a global noise level
    noise ~ gamma_bounded_below(1, 1, 0.01)
    
    # Compute the covariance between every pair (xs[i], xs[j])
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    
    # Sample from the GP using a multivariate normal distribution with
    # the kernel-derived covariance matrix.
    ys ~ mvnormal(zeros(length(xs)), cov_matrix)
    
    # Return the covariance function, for easy printing.
    return covariance_fn
end;


function serialize_trace(tr, xmin, xmax)
    (xs,) = get_args(tr)
    curveXs = collect(Float64, range(xmin, length=100, stop=xmax))
    curveYs = [predict_ys(get_retval(tr), 0.00001, xs, tr[:ys],curveXs) for i=1:5]
    Dict("y-coords" => tr[:ys], "curveXs" => curveXs, "curveYs" => curveYs)
end


tree_types = ["AllFree", "OneFree", "Connected", "SharedParent"]
@dist choose_tree_type() = tree_types[categorical([0.25, 0.25, 0.25, 0.25])]

                                          
                                          
                                          
# for iter=1:20
#     (tr, _) = generate(cv_generation_model, (collect(Float64, -1:0.1:1),))
#     serialize_trace(tr, -5, 5)
# end



                                          
                                          

# struct XYInit
#     x::Float64
#     y::Float64
# end

# struct VelocityVector
#     x::Float64
#     y::Float64
# end

# abstract type Node end

# struct InternalNode <: Node
#     left::Node
#     right::Node
#     xy::XYInit
#     vvector::VelocityVector
# end

# struct LeafNode <: Node
#     xy::XYInit
#     vvector::VelocityVector
# end

                                          

