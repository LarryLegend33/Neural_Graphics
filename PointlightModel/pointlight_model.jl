using Makie
using AbstractPlotting
using MakieLayout
using Gen
using LinearAlgebra
using LightGraphs
using MetaGraphs
using Random
#using TikzGraphs

#- One main question is whether we are going to try to reconstruct the identity after the fact. I.e. Are the xs and ys completely known in time and space? We can do simultaneous inference on x and y values wrt t. Can also do sequential monte carlo. 

#- starting with init positions b/c this is the type of custom proposal you will get from the tectum. you won't get offests for free. this model accounts for distance effects and velocity effects by traversing the tree. but it's lousy b/c it is static at 3 dots and merge / split opportunities are slim.

#-PASS THIS FUNCTION A SET OF TIMES AND AN EMPTY METADIGRAPH TO START - MetaDiGraph()
#- Involution will be you remove the edge and subtract the velocity, sample a uniform XY.
#- New edge will be you add the velocity and sample from a gaussian.

# RENDERING A REALLY NICE GRAPH:



@dist function beta_peak(μ)
    σ = √(μ*(1-μ)) / 2
    α = (((1-μ) / σ^2) - (1/μ))*(μ^2)
    β = α*((1/μ) - 1)
    beta(α, β)
end    

@gen function add_node(motion_tree::MetaDiGraph{Int64, Float64}, 
                       candidate_parents::Array{Int64, 1})
    if isempty(candidate_parents)
        add_vertex!(motion_tree)
        return motion_tree
    end
    parent_dot = first(candidate_parents)
    if isempty(inneighbors(motion_tree, parent_dot))
        parent_to_node = {(:parent, parent_dot, nv(motion_tree) + 1)} ~ bernoulli(.5)
    else
        parent_to_node = {(:parent, parent_dot, nv(motion_tree) + 1)} ~ bernoulli(.2)
    end

    if parent_to_node
        add_vertex!(motion_tree)
        add_edge!(motion_tree, parent_dot, nv(motion_tree))
        return motion_tree
    else
        {*} ~ add_node(motion_tree, filter(λ -> λ != parent_dot, candidate_parents))
    end
end    


@gen function assign_positions_and_velocities(motion_tree::MetaDiGraph{Int64, Float64},
                                              dot::Int64, ts::Array{Float64})
    if dot > nv(motion_tree)
        return motion_tree
    else
        parent = inneighbors(motion_tree, dot)
        if isempty(parent)
            xinit = {(:xinit, dot)} ~ uniform(0, 1)
            yinit = {(:yinit, dot)} ~ uniform(0, 1)
        else
            parent_position = props(motion_tree, parent[1])[:Position]
            parent_velocity_x = props(motion_tree, parent[1])[:Velocity_X]
            parent_velocity_y = props(motion_tree, parent[1])[:Velocity_Y]
            xinit = {(:xinit, dot)} ~ beta_peak(parent_position[1])
            yinit = {(:yinit, dot)} ~ beta_peak(parent_position[2])
        end
        cov_func_x = {(:cov_tree_x, dot)} ~ covariance_prior()
        cov_func_y = {(:cov_tree_y, dot)} ~ covariance_prior()
        noise = {(:noise, dot)} ~  gamma_bounded_below(1, 1, 0.01)
        covmat_x = compute_cov_matrix_vectorized(cov_func_x, noise, ts)
        covmat_y = compute_cov_matrix_vectorized(cov_func_y, noise, ts)
        x_vel = {(:x_vel, dot)} ~ mvnormal(zeros(length(ts)), covmat_x)
        y_vel = {(:y_vel, dot)} ~ mvnormal(zeros(length(ts)), covmat_y)
        if !isempty(parent)
            x_vel += parent_velocity_x
            y_vel += parent_velocity_y
        end
        # Sample from the GP using a multivariate normal distribution with
        # the kernel-derived covariance matrix.
        set_props!(motion_tree, dot,
                   Dict(:Position=>[xinit, yinit], :Velocity_X=>x_vel, :Velocity_Y=>y_vel))
        {*} ~ assign_positions_and_velocities(motion_tree, dot+1, ts)
    end
end    
            
        

@gen function generate_dotmotion(ts::Array{Float64}, 
                            motion_tree::MetaGraph{Int64, Float64},
                            num_dots::Int64)
    if nv(motion_tree) < num_dots
        motion_tree_updated = {*} ~ add_node(motion_tree, shuffle(vertices(motion_tree)))
        {*} ~ generate_dotmotion(ts, motion_tree_updated, num_dots)
    else
        motion_tree_assigned = {*} ~ assign_positions_and_velocities(motion_tree, 1, ts)
        return motion_tree_assigned
    end
end    
    


"""Create Makie Rendering Environment"""

function tree_to_coords(tree::MetaDiGraph{Int64, Float64})
    num_dots = nv(tree)
    dotmotion = fill(zeros(2), num_dots, size(props(tree, 1)[:Velocity_X])[1])
    framerate = 60
    # Assign first dot positions based on its initial XY position and velocities
    for dot in 1:num_dots
        dot_data = props(tree, dot)
        dotmotion[dot, :] = [[x, y] for (x, y) in zip(
            dot_data[:Position][1] .+ cumsum(dot_data[:Velocity_X] ./ framerate),
            dot_data[:Position][2] .+ cumsum(dot_data[:Velocity_Y] ./ framerate))]
    end

# FIX THIS IS ONLY FOR 3 DOTS
    
    dotmotion_tuples = [[Tuple(dotmotion[i, j]) for i in 1:num_dots] for j in 1:size(dotmotion)[2]]
    return dotmotion_tuples
end

function visualize_graph(motion_tree::MetaDiGraph{Int64, Float64})
    g = TikzGraphs.plot(motion_tree.graph, options="scale=10");
    TikzPictures.save(PDF("test"), g);
    graphimage = load("test.pdf");
    rot_image = imrotate(graphimage, π/2);
    return rot_image.parent
end    


function render_simulation(num_dots::Int64)
    res = 1000
    framerate = 60
    time_duration = 3
    outer_padding = 0
    num_updates = 180
    n_rows = 1
    n_cols = 3
    scene, layout = layoutscene(outer_padding,
                                resolution = (3*res, res), 
                                backgroundcolor=RGBf0(0, 0, 0))
    ts = range(1, stop=time_duration, length=time_duration*framerate)
    trace = Gen.simulate(generate_dotmotion, (convert(Array{Float64}, ts), MetaDiGraph(), num_dots))
    motion_tree = get_retval(trace)
    graph_image = visualize_graph(motion_tree)
    dotmotion = tree_to_coords(motion_tree)
    time_node = Node(1);
    f(t, coords) = coords[t]
    axes = [LAxis(scene, backgroundcolor=RGBf0(50, 50, 50)) for i in 1:n_rows, j in 1:n_cols]
    layout[1:n_rows, 1:n_cols] = axes
    time_node = Node(1);
    f(t, coords) = coords[t]
    scatter!(axes[1], lift(t -> f(t, dotmotion), time_node), markersize=20px, color=RGBf0(255, 255, 255))
    limits!(axes[1], BBox(0, 1, 0, 1))
    scatter!(axes[2], lift(t -> f(t, dotmotion), time_node), markersize=20px, color=RGBf0(255, 255, 255))
    limits!(axes[2], BBox(0, 1, 0, 1))
    display(scene)
    image!(axes[3], graph_image) 
    display(scene)
    for i in 1:num_updates
        time_node[] = i
        sleep(1/framerate)
    end
    return trace
end    

function plot_motiontree(g::Array{Int64, 2})

    graphplot(g,
          x=[0,-1/tan(π/3),1/tan(π/3)], y=[1,0,0],
          nodeshape=:circle, nodesize=1.1,
          axis_buffer=0.6,
          curves=false,
          color=:black,
          linewidth=10)
end    
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

function eval_cov_mat(node::Constant, ts::Array{Float64})
    n = length(ts)
    fill(node.param, (n, n))
end

"""Linear kernel"""
struct Linear <: PrimitiveKernel
    param::Float64
end

eval_cov(node::Linear, t1, t2) = (t1 - node.param) * (t2 - node.param)

function eval_cov_mat(node::Linear, ts::Array{Float64})
    ts_minus_param = ts .- node.param
    ts_minus_param * ts_minus_param'
end

"""Squared exponential kernel"""
struct SquaredExponential <: PrimitiveKernel
    length_scale::Float64
end

eval_cov(node::SquaredExponential, t1, t2) =
    exp(-0.5 * (t1 - t2) * (t1 - t2) / node.length_scale)

function eval_cov_mat(node::SquaredExponential, ts::Array{Float64})
    diff = ts .- ts'
    exp.(-0.5 .* diff .* diff ./ node.length_scale)
end

"""Periodic kernel"""
struct Periodic <: PrimitiveKernel
    scale::Float64
    period::Float64
end

function eval_cov(node::Periodic, t1, t2)
    freq = 2 * pi / node.period
    exp((-1/node.scale) * (sin(freq * abs(t1 - t2)))^2)
end

function eval_cov_mat(node::Periodic, ts::Array{Float64})
    freq = 2 * pi / node.period
    abs_diff = abs.(ts .- ts')
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

function eval_cov(node::Plus, t1, t2)
    eval_cov(node.left, t1, t2) + eval_cov(node.right, t1, t2)
end

function eval_cov_mat(node::Plus, ts::Vector{Float64})
    eval_cov_mat(node.left, ts) .+ eval_cov_mat(node.right, ts)
end


"""Times node"""
struct Times <: CompositeKernel
    left::Kernel
    right::Kernel
    size::Int
end

Times(left, right) = Times(left, right, size(left) + size(right) + 1)

function eval_cov(node::Times, t1, t2)
    eval_cov(node.left, t1, t2) * eval_cov(node.right, t1, t2)
end

function eval_cov_mat(node::Times, ts::Vector{Float64})
    eval_cov_mat(node.left, ts) .* eval_cov_mat(node.right, ts)
end


#-THE COVARIANCE MATRIX WILL HAVE THE DIMENSIONS OF YOUR TIME SERIES IN X, AND DEFINES THE RELATIONSHIPS BETWEEN EACH TIMEPOINT. 

"""Compute covariance matrix by evaluating function on each pair of inputs."""
function compute_cov_matrix(covariance_fn::Kernel, noise, ts)
    n = length(ts)
    cov_matrix = Matrix{Float64}(undef, n, n)
    for i=1:n
        for j=1:n
            cov_matrix[i, j] = eval_cov(covariance_fn, ts[i], ts[j])
        end
        cov_matrix[i, i] += noise
    end
    return cov_matrix
end


"""Compute covariance function by recursively computing covariance matrices."""
function compute_cov_matrix_vectorized(covariance_fn, noise, ts)
    n = length(ts)
    eval_cov_mat(covariance_fn, ts) + Matrix(noise * LinearAlgebra.I, n, n)
end

"""
Computes the conditional mean and covariance of a Gaussian process with prior mean zero
and prior covariance function `covariance_fn`, conditioned on noisy observations
`Normal(f(xs), noise * I) = ys`, evaluated at the points `new_xs`.
"""
function compute_predictive(covariance_fn::Kernel, noise::Float64,
                            ts::Vector{Float64}, pos::Vector{Float64},
                            new_xs::Vector{Float64})
    n_prev = length(ts)
    n_new = length(new_ts)
    means = zeros(n_prev + n_new)
#    cov_matrix = compute_cov_matrix(covariance_fn, noise, vcat(xs, new_xs))
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, vcat(ts, new_ts))
    cov_matrix_11 = cov_matrix[1:n_prev, 1:n_prev]
    cov_matrix_22 = cov_matrix[n_prev+1:n_prev+n_new, n_prev+1:n_prev+n_new]
    cov_matrix_12 = cov_matrix[1:n_prev, n_prev+1:n_prev+n_new]
    cov_matrix_21 = cov_matrix[n_prev+1:n_prev+n_new, 1:n_prev]
    @assert cov_matrix_12 == cov_matrix_21'
    mu1 = means[1:n_prev]
    mu2 = means[n_prev+1:n_prev+n_new]
    conditional_mu = mu2 + cov_matrix_21 * (cov_matrix_11 \ (pos - mu1))
    conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 * (cov_matrix_11 \ cov_matrix_12)
    conditional_cov_matrix = 0.5 * conditional_cov_matrix + 0.5 * conditional_cov_matrix'
    (conditional_mu, conditional_cov_matrix)
end

"""
Predict output values for some new input values
"""
function predict_pos(covariance_fn::Kernel, noise::Float64,
                     ts::Vector{Float64}, pos::Vector{Float64},
                     new_ts::Vector{Float64})
    (conditional_mu, conditional_cov_matrix) = compute_predictive(
        covariance_fn, noise, ts, pos, new_ts)
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
@gen function cv_generation_model(ts::Vector{Float64})
    
    # Generate a covariance kernel
    covariance_fn = { :tree } ~ covariance_prior()
    
    # Sample a global noise level
    #    noise ~ gamma_bounded_below(.01, .01, 0.01)
    noise = 0
    
    # Compute the covariance between every pair (xs[i], xs[j])
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, ts)
    
    # Sample from the GP using a multivariate normal distribution with
    # the kernel-derived covariance matrix.
    ys ~ mvnormal(zeros(length(ts)), cov_matrix)
    
    # Return the covariance function, for easy printing.
    return covariance_fn
end;


function serialize_trace(tr, tmin, tmax)
    (ts,) = get_args(tr)
    curveT = collect(Float64, range(tmin, length=100, stop=tmax))
    curvePos = [predict_pos(get_retval(tr), 0.00001, ts, tr[:pos],curveT) for i=1:5]
    Dict("y-coords" => tr[:pos], "curveT" => curveT, "curvePos" => curvePos)
end


tree_types = ["AllFree", "OneFree", "Connected", "SharedParent"]
@dist choose_tree_type() = tree_types[categorical([0.25, 0.25, 0.25, 0.25])]

                                          
                                          

                                          

