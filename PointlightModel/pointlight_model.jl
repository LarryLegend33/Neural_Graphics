using Makie
using AbstractPlotting
using MakieLayout
using Gen
using LinearAlgebra
using LightGraphs
using MetaGraphs
using Random
using Images
using TikzGraphs
using TikzPictures
using ShiftedArrays
using ColorSchemes
using Statistics

#- One main question is whether we are going to try to reconstruct the identity after the fact. I.e. Are the xs and ys completely known in time and space? We can do simultaneous inference on x and y values wrt t. Can also do sequential monte carlo. 

#- starting with init positions b/c this is the type of custom proposal you will get from the tectum. you won't get offests for free. this model accounts for distance effects and velocity effects by traversing the tree. 

#- One thing you might want to think about is keeping the same exact structure but resampling the timeseries. If you do this, may be a good way to test good choices in structure vs sample. 

@dist function beta_peak(μ)
    σ = √(μ*(1-μ)) / 2
    α = (((1-μ) / σ^2) - (1/μ))*(μ^2)
    β = α*((1/μ) - 1)
    beta(α, β) 
end

# Make sure populate edges generates any possible edge combination 

@gen function populate_edges(motion_tree::MetaDiGraph{Int64, Float64},
                             candidate_parents::Array{Int64, 1},
                             current_dot::Int64)
    if isempty(candidate_parents)
        return motion_tree
    end
    cand_parent = first(candidate_parents)
    if has_edge(motion_tree, current_dot, cand_parent) || ne(motion_tree) == nv(motion_tree) - 1 
        add_edge = { (:edge, cand_parent, current_dot) } ~  bernoulli(0)
    else
        if isempty(inneighbors(motion_tree, cand_parent))
            add_edge = { (:edge, cand_parent, current_dot) } ~  bernoulli(.5)
        else
            add_edge = { (:edge, cand_parent, current_dot) } ~  bernoulli(.2)
        end
    end
    if add_edge
        add_edge!(motion_tree, cand_parent, current_dot)
    end
    {*} ~ populate_edges(motion_tree, candidate_parents[2:end], current_dot)
end


@gen function generate_dotmotion(ts::Array{Float64}, 
                                 motion_tree::MetaDiGraph{Int64, Float64},
                                 candidate_children::Array{Int64, 1})
    if !isempty(candidate_children)
        current_dot = first(candidate_children)
        candidate_parents = shuffle(filter(λ -> λ != current_dot, vertices(motion_tree)))
        motion_tree_updated = {*} ~ populate_edges(motion_tree, candidate_parents, current_dot)
        {*} ~ generate_dotmotion(ts,
                                 motion_tree_updated,
                                 candidate_children[2:end])
    else

        # order the vertices by number of incoming edges.
        dot_list = sort(collect(1:nv(motion_tree)), by=(ϕ->size(inneighbors(motion_tree, ϕ))))
        motion_tree_assigned = {*} ~ assign_positions_and_velocities(motion_tree,
                                                                     dot_list,
                                                                     ts)
        return motion_tree_assigned
    end
end    


# see if motion tree is a traced variable. if so, it will have a score. 

@gen function assign_positions_and_velocities(motion_tree::MetaDiGraph{Int64, Float64},
                                              dots::Array{Int64}, ts::Array{Float64})
    position_var = 1
    if isempty(dots)
        return motion_tree
    else
        dot = first(dots)
        parents = inneighbors(motion_tree, dot)
        start_x = 5
        if isempty(parents)
            #            start_x = {(:start_x, dot)} ~ uniform_discrete(3, 7)
            start_y = {(:start_y, dot)} ~ uniform_discrete(2, 8)
        else
            if size(parents)[1] > 1
                avg_parent_position = mean([props(motion_tree, p)[:Position] for p in parents])
                parent_position = [round(Int, pp) for pp in avg_parent_position]
            else
                parent_position = props(motion_tree, parents[1])[:Position]
            end
          #  start_x = {(:start_x, dot)} ~ normal(parent_position[1], position_var)
          #  start_y = {(:start_y, dot)} ~ normal(parent_position[2], position_var)
#            start_x = {(:start_x, dot)} ~ uniform_discrete(parent_position[1]-1, parent_position[1]+1)
            start_y = {(:start_y, dot)} ~ uniform_discrete(parent_position[2]-1, parent_position[2]+1)

            parent_velocities_x = [props(motion_tree, p)[:Velocity_X] for p in parents]
            parent_velocities_y = [props(motion_tree, p)[:Velocity_Y] for p in parents]
        end

#        cov_func_x = {(:cov_tree_x, dot)} ~ covariance_prior()
#        cov_func_y = {(:cov_tree_y, dot)} ~ covariance_prior(typeof(cov_func_x))
#        noise = {(:noise, dot)} ~  gamma_bounded_below(1, 1, 0.01)
        #        cov_func = {(:cov_tree, dot)} ~ covariance_prior()
        #        cov_func = {(:cov_tree, dot)} ~ covariance_simple()
        cov_func = {*} ~ covariance_simple(dot)
#        flip_y = { (:flip_y, dot) } ~ bernoulli(.5)
        noise = 0.001
        covmat_x = compute_cov_matrix_vectorized(cov_func, noise, ts)
        covmat_y = compute_cov_matrix_vectorized(cov_func, noise, ts)
        x_vel = {(:x_vel, dot)} ~ mvnormal(zeros(length(ts)), covmat_x) 
        #   y_vel = {(:y_vel, dot)} ~ mvnormal(zeros(length(ts)), covmat_y)
        y_vel = [0 for xv in x_vel]
        if !isempty(parents)
            if size(parents)[1] == 1
                x_vel += parent_velocities_x[1]
                y_vel += parent_velocities_y[1]
            else
                x_vel += sum(parent_velocities_x)
                y_vel += sum(parent_velocities_y)
            end
        end
        # Sample from the GP using a multivariate normal distribution with
        # the kernel-derived covariance matrix.
        set_props!(motion_tree, dot,
                   Dict(:Position=>[start_x, start_y], :Velocity_X=>x_vel, :Velocity_Y=>y_vel))
        {*} ~ assign_positions_and_velocities(motion_tree, dots[2:end], ts)
    end
end    

#start with this just being the simulated data itself. eventually have it be a biophysical implementation of a
# tectal map

# function neural_detector()
#     neural_constraints = Gen.choicemap()
#     neural_constraints[:start_x] = 0.1
#     neural_constraints[:start_y] = 0.1

    
# end


function force_assign_dotpositions(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    #this function will force a particular set of observed positions to start_x and start_y of
    # a specific dot. 
end    


# make this able to take various lenghts of ts and update with SMC
        
function enumerate_possibilities(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    num_dots = nv(get_retval(trace))
    enum_constraints = Gen.choicemap()
    kernel_combos = [kernel_types for i in 1:num_dots]
    kernel_choices = collect(Iterators.product(kernel_combos...))
    possible_edges = [e for e in Iterators.product(1:num_dots, 1:num_dots) if e[1] != e[2]]
    truth_entry = [[0,1] for i in 1:size(possible_edges)[1]]
    edge_truthtable = [j for j in Iterators.product(truth_entry...) if sum(j) < num_dots]
    scores = []
    for eg in edge_truthtable
        for (eg_id, e) in enumerate(eg)
            if e == 1
              enum_constraints[(:edge, possible_edges[eg_id][1], possible_edges[eg_id][2])] = true
            else
              enum_constraints[(:edge, possible_edges[eg_id][1], possible_edges[eg_id][2])] = false
            end
        end
        for kc in kernel_choices
            for (dot, k) in enumerate(kc)
                enum_constraints[(:kernel_type, dot)] = k
            end
            (new_trace, weight, a, ad) = Gen.update(trace, get_args(trace), (NoChange(),), enum_constraints)
            append!(scores, weight)
        end
    end
    score_matrix = reshape(scores, prod(collect(size(kernel_choices))), size(edge_truthtable)[1])
    return score_matrix, kernel_choices, edge_truthtable
end    
    
function plot_heatmap(score_matrix::Array{Any, 2}, kernels, edge_truth)
    scene, layout = layoutscene(resolution=(1200,900))
    axes = [LAxis(scene)]
    heatmap!(axes[1], score_matrix, colormap=:thermal)
    layout[1,1] = axes[1]
    axes[1].xticks = (1:prod(collect(size(kernels))), [string(k) for k in kernels])
    axes[1].yticks = (1:size(edge_truth)[1], [string(e) for e in edge_truth])
    display(scene)
end    

    # possible_edges is an array of all possible connections between dots. truth table
    # dictates whether the edge exists or not in the current proposed tree
    
    # will be 24 possible arrangements of velocity for each tree. 
        
        
    
    # take the trace into this function and use Gen.update to change the choicemap.
    
    

                         
    # have to cycle through all possible states. first make it simple. 4 possible motion choices, xcoord only changing
    # then each possible scene graph. 
    



    


"""Create Makie Rendering Environment"""

function tree_to_coords(tree::MetaDiGraph{Int64, Float64},
                        framerate::Int64)
    num_dots = nv(tree)
    dotmotion = fill(zeros(2), num_dots, size(props(tree, 1)[:Velocity_X])[1])
    # Assign first dot positions based on its initial XY position and velocities
    for dot in 1:num_dots
        dot_data = props(tree, dot)
        dotmotion[dot, :] = [[x, y] for (x, y) in zip(
            dot_data[:Position][1] .+ cumsum(dot_data[:Velocity_X] ./ framerate),
            dot_data[:Position][2] .+ cumsum(dot_data[:Velocity_Y] ./ framerate))]
    end
    dotmotion_tuples = [[Tuple(dotmotion[i, j]) for i in 1:num_dots] for j in 1:size(dotmotion)[2]]
    return dotmotion_tuples
end

function visualize_graph(motion_tree::MetaDiGraph{Int64, Float64},
                         resolution::Int64)
    g = TikzGraphs.plot(motion_tree.graph,
                        edge_style="yellow", 
                        node_style="draw, rounded corners, fill=blue!20",
                        options="scale=8, font=\\huge\\sf");
    TikzPictures.save(PDF("test"), g);
    graphimage = load("test.pdf");
    rot_image = imrotate(graphimage, π/2);
    scale_ratio = (resolution*.75) / maximum(size(rot_image))
    resized_image = imresize(rot_image, ratio=scale_ratio)
    return resized_image
end    

function render_simulation(num_dots::Int64)
    framerate = 60
    bounds = 10
    time_duration = 10
    res = 850
    outer_padding = 0
    num_updates = framerate * time_duration
    ts = range(1, stop=time_duration, length=time_duration*framerate)
    trace = Gen.simulate(generate_dotmotion, (convert(Array{Float64}, ts),
                                              MetaDiGraph(num_dots), shuffle(1:num_dots)))

    motion_tree = get_retval(trace)
    graph_image = visualize_graph(motion_tree, res)
    dotmotion = tree_to_coords(motion_tree, framerate)
    f(t, coords) = coords[t]

    n_rows = 1
    n_cols = 3
    scene, layout = layoutscene(outer_padding,
                                resolution = (3*res, res), 
                                backgroundcolor=RGBf0(0, 0, 0))
    
    axes = [LAxis(scene, backgroundcolor=RGBf0(0, 0, 0)) for i in 1:n_rows, j in 1:n_cols]
    layout[1:n_rows, 1:n_cols] = axes
    time_node = Node(1);
    f(t, coords) = coords[t]
    scatter!(axes[1], lift(t -> f(t, dotmotion), time_node), markersize=10px, color=RGBf0(255, 255, 255))
    limits!(axes[1], BBox(0, bounds, 0, bounds))
    scatter!(axes[2], lift(t -> f(t, dotmotion), time_node), markersize=10px, color=RGBf0(255, 255, 255))
    limits!(axes[2], BBox(0, bounds, 0, bounds))
    image!(axes[3], graph_image)
    limits!(axes[3], BBox(0, res, 0, res))
    for j in 1:num_dots
        println(trace[(:kernel_type, j)])
#        println(trace[(:cov_tree_y, j)])
    end
    display(scene)
    for i in 1:num_updates
        time_node[] = i
        sleep(1/framerate)
    end
    return trace
end    

function render_simulation(num_dots::Int64,
                           trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    framerate = 60
    bounds = 10
    time_duration = 10
    res = 850
    outer_padding = 0
    num_updates = framerate * time_duration
    ts = range(1, stop=time_duration, length=time_duration*framerate)
    motion_tree = get_retval(trace)
    graph_image = visualize_graph(motion_tree, res)
    dotmotion = tree_to_coords(motion_tree, framerate)
    f(t, coords) = coords[t]
    n_rows = 1
    n_cols = 3
    scene, layout = layoutscene(outer_padding,
                                resolution = (3*res, res), 
                                backgroundcolor=RGBf0(0, 0, 0))
    
    axes = [LAxis(scene, backgroundcolor=RGBf0(0, 0, 0)) for i in 1:n_rows, j in 1:n_cols]
    layout[1:n_rows, 1:n_cols] = axes
    time_node = Node(1);
    f(t, coords) = coords[t]
    scatter!(axes[1], lift(t -> f(t, dotmotion), time_node), markersize=10px, color=RGBf0(255, 255, 255))
    limits!(axes[1], BBox(0, bounds, 0, bounds))
    scatter!(axes[2], lift(t -> f(t, dotmotion), time_node), markersize=10px, color=RGBf0(255, 255, 255))
    limits!(axes[2], BBox(0, bounds, 0, bounds))
    image!(axes[3], graph_image)
    limits!(axes[3], BBox(0, res, 0, res))
    display(scene)
    for i in 1:num_updates
        time_node[] = i
        sleep(1/framerate)
    end
    return trace
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

"""Random Walk Kernel"""
struct RandomWalk <: PrimitiveKernel
    param::Float64
end

function eval_cov(node::RandomWalk, t1, t2)
    if t1 == t2
        node.param
    else
        0
    end
end        

function eval_cov_mat(node::RandomWalk, ts::Array{Float64})
    n = length(ts)
    Diagonal(node.param * ones(n))
end

    
"""Constant kernel"""
struct Constant <: PrimitiveKernel
    param::Float64
end

eval_cov(node::Constant, t1, t2) = node.param


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

# This is an array of data types. Each data type takes a parameter, and each data type has a multiple dispatch
# call associated with it to create a covariance matrix. 
    
#kernel_types = [RandomWalk, Constant, Linear, Periodic]
#@dist choose_kernel_type() = kernel_types[categorical([.25, .25, .25, .25])]

kernel_types = [RandomWalk, Constant, Periodic]
@dist choose_kernel_type() = kernel_types[categorical([.3, .4, .3])]


@gen function covariance_simple(kt)
    kernel_type = {(:kernel_type, kt)} ~ choose_kernel_type()
    if kernel_type == Periodic
        kernel_args = [.5, 5]
    elseif kernel_type == Constant
        kernel_args = [3]
    elseif kernel_type == Linear
        kernel_args = [.2]
    elseif kernel_type == RandomWalk
        kernel_args = [2]
    else
        kernel_args = [1]
    end
    return kernel_type(kernel_args...)
end 


@gen function covariance_prior()
    # Choose a type of kernel
    kernel_type = { :kernel_type } ~ choose_kernel_type()
    # If this is a composite node, recursively generate subtrees. For now, too complex. 
    if in(kernel_type, [Plus, Times])
        return kernel_type({ :left } ~ covariance_prior(), { :right } ~ covariance_prior())
    end
    # Otherwise, generate parameters for the primitive kernel.
    if kernel_type == Periodic
        kernel_args = [{ :scale } ~ uniform(0, 1), { :period } ~ uniform(0, 10)]
    elseif kernel_type == Constant
        kernel_args = [{ :param } ~ uniform(0, 3)]
    elseif kernel_type == Linear
        kernel_args = [{ :param } ~ uniform(0, 1)]
    elseif kernel_type == RandomWalk
        kernel_args = [{ :param } ~ uniform(0, 10)]
    else
        kernel_args = [{ :param } ~ uniform(0, 1)]
    end
    return kernel_type(kernel_args...)
end

@dist gamma_bounded_below(shape, scale, bound) = gamma(shape, scale) + bound
                                          








                                          

