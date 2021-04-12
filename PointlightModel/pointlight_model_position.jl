using GLMakie
using Gen
using GenGridEnumeration
using OrderedCollections
using LinearAlgebra
using LightGraphs
using MetaGraphs
using Random
#using Images
using ColorSchemes
using Statistics
using StatsBase
using CurricularAnalytics
using BSON: @save, @load
using MappedArrays
using Base.Threads: @spawn
using SpecialFunctions: loggamma


# have to kill old threads when you launch a new one, because render thread keeps running.
# try first limiting the total number of threads.

# g = UniformPointPushforwardGrid(trace, OrderedDict(
#     (:edge, 1, 2) => DiscreteSingletons([true, false])))

# g = UniformPointPushforwardGrid(trace, OrderedDict(
#     :num_dots => DiscreteSingletons([1,2,3])))

# code for 1 on top 
        # for parent in 1:dot-1
        #     add_edge = {(:edge, parent, dot)} ~ bernoulli(edge_prob)
        #     if add_edge
        #         add_edge!(motion_graph, parent, dot)
        #         edge_prob = 0
        #     end
        # end

# Assign invisibility at the end. Any nodes with no children can't be invisible.

# Main problem is the periodic kernel is less constrained than linear. Linear will basically fit, while
# the periodic kernel is much more powerful and can generate more variable motion patterns.
# counting the peak to peak amplitude and using heuristics to get period may be helpful.
# Make sure tomorrow that you know how period translates to amount of repeats...this wasn't completely clear before. 


# Possible Proposals that fit: measure the consistency of xy magnitude from each dot.
# if XY magnitude is basically constant, this explains fully inherited patterns (i.e. linear->stationary)
# the wheel stimulus;
# dot moving inside the bar is activating TP neurons in ewert -- it's an antiworm stimulus!
# what is evidence for offset of a 

# NOTE THAT NOISE WILL DISRUPT EVERYTHING. THATS THE WHOLE POINT.
# NOISE WILL SCREW UP HEURISTICS, MAKE TIMESERIES CONSTRAINING DIFFICULT BECAUSE CORRELATIONS WILL
# ARISE FROM NOISE, ETC. JITTER MAY ALSO HELP! SHARING MORE VELOCITY WILL BE BETTER.
# PROPOSING THE DIRECT TIMESERIES OF CHILD NODES I DONT THINK HURTS...

# No


# invisible leafs would seem a priori weird, but you imagine what a wrist looks like on the end of a hand. or
# foot at the end of a leg. 
# when we see an arm we can fill in the rest of the human. some torso dot / leg dot is controlling the forward
# translation of the arm.

# 


""" GENERATIVE CODE: These functions output dotmotion stimuli using GP-generated timeseries"""

@gen function generate_dot_scene(ts::Array{Float64})
    num_visible = { :num_visible } ~ poisson_bounded_below(1, 1)
    num_dots = { :num_dots } ~ poisson_bounded_below(.2, num_visible)
    motion_graph = MetaDiGraph(num_dots)
    dperm1 = { :dp1 } ~ randomPartialPermutation(num_dots, num_dots)
    dperm2 = { :dp2 } ~ randomPartialPermutation(num_dots, num_dots)
    candidate_edges = [(n1, n2) for n1 in dperm1 for n2 in dperm2 if n1!=n2]
    vis_or_invis = [d <= num_visible ? {(:isvisible, d)} ~ bernoulli(1) : {(:isvisible, d)} ~ bernoulli(0) for d in 1:num_dots]
    assignment_order = {*} ~ populate_edges(motion_graph, candidate_edges)
    for dot in assignment_order
        cov_func_x, cov_func_y = { (:cf_tree, dot) } ~ covfunc_prior([.4, 0.0, .4, .2], dot, 0)
        covmat_x = compute_cov_matrix_vectorized(cov_func_x, ϵ, ts)
        covmat_y = compute_cov_matrix_vectorized(cov_func_y, ϵ, ts)
        start_x = {(:start_x, dot)} ~ uniform(-25, 25)
        start_y = {(:start_y, dot)} ~ uniform(-25, 25)
        x_timeseries = {(:x_timeseries, dot)} ~ mvnormal(zeros(length(ts)), covmat_x)
        y_timeseries = {(:y_timeseries, dot)} ~ mvnormal(zeros(length(ts)), covmat_y)
        perceptual_noise_magnitude = {(:perceptual_noise_magnitude, dot)} ~ multinomial(param_dict[:perceptual_noise_magnitude])
        noise = {*} ~ generate_white_noise(perceptual_noise_magnitude, :perceptual_noise, ts, dot)
        jitter_magnitude = {(:jitter_magnitude, dot)} ~ multinomial(param_dict[:jitter_magnitude])
        jitter = {*} ~ generate_white_noise(float(jitter_magnitude), :jitter, ts, dot)
        dot_dict = Dict(:jitter => jitter,
                        :x_timeseries => x_timeseries,
                        :y_timeseries => y_timeseries,
                        :perceptual_noise => noise,
                        :isvisible => vis_or_invis[dot],
                        :MType => mtype_extractor(cov_func_x))
        set_props!(motion_graph, dot, dot_dict)
        output_covmat = compute_cov_matrix_vectorized(RandomWalk(0), ϵ, ts)
        parent_x, parent_y = get_parent_influence(motion_graph, dot, ts)
        x_obs, y_obs = create_observable(x_timeseries, y_timeseries, jitter, noise, start_x, start_y,
                                         parent_x, parent_y)
        x_observable = {(:x_observable, dot)} ~ mvnormal(x_obs, output_covmat)
        y_observable = {(:y_observable, dot)} ~ mvnormal(y_obs, output_covmat)
    end
    
    return motion_graph
end


@gen function generate_white_noise(variance::Float64, rv::Symbol, ts::Array{Float64}, dot::Int64)
    covmat = compute_cov_matrix_vectorized(RandomWalk(variance), ϵ, ts)
    white_noise_x = {(rv, dot, :x)} ~ mvnormal(zeros(length(ts)), covmat)
    white_noise_y = {(rv, dot, :y)} ~ mvnormal(zeros(length(ts)), covmat)
    return white_noise_x, white_noise_y
end


@gen function populate_edges(motion_graph::MetaDiGraph{Int64, Float64},
                             candidate_pairs::Array{Tuple{Int64,Int64},1})
    if isempty(candidate_pairs)
        return reverse(sort(collect(1:nv(motion_graph)), by=λ -> length(longest_path(motion_graph, λ))))
    end
    (current_dot, cand_parent) = first(candidate_pairs)
    current_paths = all_paths(motion_graph)
    # prevents recursion in the motion tree. can add this back in someday to model non-rigid bodies
    # like a ring of beads
    dot_has_parent = !isempty(inneighbors(motion_graph, current_dot))
    path_recurrence = any([current_dot in path && cand_parent in path for path in current_paths])
    if path_recurrence || dot_has_parent
        add_edge = { (:edge, cand_parent, current_dot) } ~  bernoulli(0)
    else
#        if isempty(inneighbors(motion_graph, cand_parent))
        add_edge = { (:edge, cand_parent, current_dot) } ~  bernoulli(.3)
        # else
        #     add_edge = { (:edge, cand_parent, current_dot) } ~  bernoulli(.1)
        # end
    end
    if add_edge
        add_edge!(motion_graph, cand_parent, current_dot)
    end
    {*} ~ populate_edges(motion_graph, candidate_pairs[2:end])
end


function create_observable(x_timeseries, y_timeseries, jitter, noise, start_x, start_y,
                           parent_x, parent_y)
    x_obs = deepcopy(x_timeseries)
    y_obs = deepcopy(y_timeseries)
    x_obs += jitter[1]
    x_obs += parent_x
    x_obs += noise[1]
    y_obs += jitter[2]
    y_obs += parent_y
    y_obs += noise[2]
    x_obs .+= (start_x - x_obs[1])
    y_obs .+= (start_y - y_obs[1])
    return x_obs, y_obs
end

# this is WRONG. have to get entire path of inneighbors, not just immediate above neighbor. 

function get_parent_influence(motion_graph, dot, ts)
    parent_dots = reachable_to(motion_graph.graph, dot)
    parent_x = zeros(length(ts))
    parent_y = zeros(length(ts))
    for parent_dot in parent_dots
        parent_x += deepcopy(props(motion_graph, parent_dot)[:x_timeseries])
        parent_x += props(motion_graph, parent_dot)[:jitter][1]
        parent_y += deepcopy(props(motion_graph, parent_dot)[:y_timeseries])
        parent_y += props(motion_graph, parent_dot)[:jitter][2]
    end
    return parent_x, parent_y
end

# proposal will be count the amount of observable variables in the choicemap. constrain the number of dots to a tight distribution on it.

# calculate as a smart proposal: slope of line, 

""" PROPOSALS AND INFERENCE WRAPPERS W PLOTTING METHODS """ 


@gen function dotnum_and_timeseries_proposal(current_trace)
    observed_dot_ids = []
    ts_covmat = compute_cov_matrix_vectorized(RandomWalk(0), ϵ*100, get_args(current_trace)[1])
    # here you get the number of dots observed for free,
    # but in the future just detect the coords of visible dots and count them.
    num_dots = { :num_dots } ~ poisson_bounded_below(.1, current_trace[:num_visible])
    new_tree = MetaDiGraph(num_dots)
    dperm1 = { :dp1 } ~ randomPartialPermutation(num_dots, num_dots)
    dperm2 = { :dp2 } ~ randomPartialPermutation(num_dots, num_dots)
    candidate_edges = [(n1, n2) for n1 in dperm1 for n2 in dperm2 if n1!=n2]
    dot_order = {*} ~ populate_edges(new_tree, candidate_edges)
    for dot in dot_order
        influence_x = zeros(length(get_args(current_trace)[1]))
        influence_y = zeros(length(get_args(current_trace)[1]))
        for parent in inneighbors(new_tree, dot)
            if parent <= current_trace[:num_visible]
                influence_x -= current_trace[(:x_observable, parent)]
                influence_y -= current_trace[(:y_observable, parent)]
            end
        end
        if dot <= current_trace[:num_visible]
            {(:x_timeseries, dot)} ~ mvnormal(current_trace[(:x_observable, dot)] + influence_x, ts_covmat)
            {(:y_timeseries, dot)} ~ mvnormal(current_trace[(:y_observable, dot)] + influence_y, ts_covmat)
        end
    end
end

    

""" THESE FUNCTIONS WRAP SAMPLING RENDERING AND INFERENCE """

# draw from prior

function dotwrap()
    constraints = choicemap()
    dotwrap(constraints)
end

function dotwrap(constraints::Gen.DynamicChoiceMap)
    (trace, weight) = Gen.generate(generate_dot_scene, 
                                   (timepoints,),  
                                   constraints)
    #    dotwrap(trace, choicemap())
    render_dotmotion(trace, true, true)
    return trace, weight
end
                 
function dotwrap(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}, constraints::Gen.DynamicChoiceMap)
    args = get_args(trace)
    (updated_trace, w, retdiff, discard) = Gen.update(trace, args, (), constraints)
    render_dotmotion(updated_trace, true, true)
#    @spawn render_dotmotion(updated_trace, false)
#    ax = visualize_scenegraph(scenegraph, get_choices(updated_trace))
    # s = Scene()
    # display(s)
    return trace, w
end    


# this is a key helper function. by executing dotwrap(make_constraints()) or dotwrap(trace, make_constraints()), 

# for what we want (a change in the x_observable w/ noise constraint changes), want to input the original trace. 
# extract the choicemap, then remove its x_observable. loop through constraints and replace the original choicemap's values
# with that of the constraints using choices[addr] = val. call Gen.generate on the new choicemap, which yields a new x_observable. 



# Form of custom proposal. Lets manipulate the cov function first.
# Can make a set of particles for this. Per particle, draw a few samples instead of
# just one. This will cover a few phases of the periodic wave. 

# param_dict = Dict(Periodic => [collect(1:10), collect(.5:.5:2), collect(.5:.25:2)],
#                   Linear => [[0, 10, 20], collect(0:5:20)],
#                   SquaredExponential => [collect(1:20), collect(.05:.05:.5)],
#                   :jitter_magnitude => [0],    # previously [.01, .1, 1]
#                   :perceptual_noise_magnitude => [0, .01])

function compare_edge_noedge(num_samples)
    fig, ax = hist([dotwrap(make_constraints_noedge())[2] for i in 1:num_samples], color=(RGBf0(0,0,0), .3))
    hist!(ax, [dotwrap(make_constraints_edge())[2] for i in 1:num_samples], color=(RGBf0(0,255,0), .3))
    display(fig)
end    


function make_constraints_edge()
    constraints = choicemap()
    bar_height = 15
    ball_freq = .15
    wheel_dictionary = Dict(
        :num_dots => 2,
        :num_visible => 2,
        (:edge, 2, 1) => true, 
        (:edge, 1, 2) => false, 
        # (:x_timeseries, 1) => [-5*cos(ball_freq*i) for i in 0:num_position_points-1],
        # (:y_timeseries, 1) => [5*sin(ball_freq*i) for i in 0:num_position_points-1],
        # (:x_timeseries, 2) => [(50/num_position_points * i) - 18 for i in 0:num_position_points-1],
        # (:y_timeseries, 2) => zeros(num_position_points),
        (:x_observable, 1) => [(50/num_position_points * i) - 18 for i in 0:num_position_points-1] + [-5*cos(ball_freq*i) for i in 0:num_position_points-1],
        (:y_observable, 1) => [5*sin(ball_freq*i) for i in 0:num_position_points-1],
        (:x_observable, 2) => [(50/num_position_points * i) - 18 for i in 0:num_position_points-1],
        (:y_observable, 2) => zeros(num_position_points),
        ((:cf_tree, 1) => :kernel_type) => Periodic,
        ((:cf_tree, 2) => :kernel_type) => Linear,
        (:start_x, 1) => -23,
        (:start_y, 1) => 0,
        (:start_x, 2) => -18,
        (:start_y, 2) => 0,
        (:perceptual_noise_magnitude, 1) => 0.0,
        (:jitter_magnitude, 1) => 0.0,
        (:perceptual_noise_magnitude, 2) => 0.0,
        (:jitter_magnitude, 2) => 0.0) 
    constraints = choicemap([Tuple(d) for d in wheel_dictionary]...)
    return constraints
end
    
function make_constraints_noedge()
    constraints = choicemap()
    bar_height = 15
    ball_freq = .15
    wheel_dictionary = Dict(
        :num_dots => 1,
        :num_visible => 1,
        (:edge, 2, 1) => false, 
        (:edge, 1, 2) => false,
        (:x_observable, 2) => [(50/num_position_points * i) - 18 for i in 0:num_position_points-1] + [-5*cos(ball_freq*i) for i in 0:num_position_points-1],
        (:y_observable, 2) => [5*sin(ball_freq*i) for i in 0:num_position_points-1],
        (:x_observable, 1) => [(50/num_position_points * i) - 18 for i in 0:num_position_points-1],
        (:y_observable, 1) => zeros(num_position_points),
        ((:cf_tree, 1) => :kernel_type) => Plus,
        ((:cf_tree, 1) => :left => :kernel_type) => Periodic,
        ((:cf_tree, 1) => :right => :kernel_type) => Linear,
        ((:cf_tree, 2) => :kernel_type) => Linear,
        (:start_x, 1) => -23,
        (:start_y, 1) => 0,
        (:start_x, 2) => -18,
        (:start_y, 2) => 0,
        (:perceptual_noise_magnitude, 1) => 0.0,
        (:jitter_magnitude, 1) => 0.0,
        (:perceptual_noise_magnitude, 2) => 0.0,
        (:jitter_magnitude, 2) => 0.0)
    constraints = choicemap([Tuple(d) for d in wheel_dictionary]...)
    return constraints
end

function make_constraints()
    constraints = choicemap()
    bar_height = 15
    ball_freq = .15
    bar_dictionary = Dict(:num_dots => 3,
                          (:kernel_type, 1) => Linear,
                          (:kernel_type, 2) => Linear,
                          (:kernel_type, 3) => Periodic, 
                          (:perceptual_noise_magnitude, 1) => 0.0,
                          (:perceptual_noise_magnitude, 2) => 0.0,
                          (:perceptual_noise_magnitude, 3) => 0.0,
                          (:jitter_magnitude, 1) => 0.0,
                          (:jitter_magnitude, 2) => 0.0,
                          (:jitter_magnitude, 3) => 0.0,
                          (:isvisible, 1) => true,
                          (:isvisible, 2) => true,
                          (:isvisible, 3) => true,
                          (:edge, 1, 2) => true,
                          (:edge, 1, 3) => true,
                          (:edge, 2, 3) => false,
                          (:x_timeseries, 1) => [(50/num_position_points * i) - 20 for i in 0:num_position_points-1],
                          (:y_timeseries, 1) => bar_height*ones(num_position_points),
                          (:x_timeseries, 2) => -20*ones(num_position_points),
                          (:y_timeseries, 2) => -bar_height*ones(num_position_points),
                          (:x_timeseries, 3) => -20*ones(num_position_points),
                          (:y_timeseries, 3) => [bar_height*sin(ball_freq*i) for i in 0:num_position_points-1])


    freestyle = (:num_dots => 1,
                 (:perceptual_noise_magnitude, 1) => 0.0,
                 (:jitter_magnitude, 1) => 0.0,
                 :num_visible => 1,
                 ((:cf_tree, 1) => :kernel_type) => Linear, 
                 (:start_x, 1) => 0,
                 (:start_y, 1) => 0,
                 ((:cf_tree, 1) => (:variance, :x)) =>  20,
                 ((:cf_tree, 1) => (:offset, :x)) =>  0,
                 ((:cf_tree, 1) => (:variance, :y)) =>  20,
                 ((:cf_tree, 1) => (:offset, :y)) =>  0)
                 

    # variance controls speed of linear motion, offset controls where it crosses axis. since all
    # lines are normalized to the start_x and y, leave offset at 0 and only switch the speed. 
    constraints = choicemap([Tuple(d) for d in bar_dictionary]...)
    return constraints
end





# get_submap will work here, but need observations to be indexed, not choicemap. 
# you'll have cf_func


   

# make this able to take various lenghts of ts and update with SMC

function loopfilter(edges, truthtab)
    filtered_truthtab = []
    for t_entry in truthtab
        edges_in_entry = [e for (e,t) in zip(edges, t_entry) if t==1]
        if !any(map(x -> (x[2], x[1]) in edges_in_entry, edges_in_entry))
            push!(filtered_truthtab, t_entry)
        end
    end
    return filtered_truthtab
end

function bayesian_observer(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    num_dots = nv(get_retval(trace)[1])
    kernel_choices = [kernel_types for i in 1:num_dots]
    jitter_choices = [param_dict[:jitter] for i in 1:num_dots]
    kernel_combos = collect(Iterators.product(kernel_choices...))
    jitter_combos = collect(Iterators.product(jitter_choices...))
    possible_edges = [(i, j) for i in 1:num_dots for j in 1:num_dots if i < j]
    truth_entry = [[0,1] for i in 1:size(possible_edges)[1]]
    if !isempty(truth_entry)
        unfiltered_truthtable = [j for j in Iterators.product(truth_entry...) if sum(j) < num_dots]
        edge_truthtable = loopfilter(possible_edges, unfiltered_truthtable)
    else
        unfiltered_truthtable = edge_truthtable = [()]
    end
    trace_args = get_args(trace)
    trace_choices = get_choices(trace)
    trace_retval = get_retval(trace)
    top_score = -Inf
    top_trace = trace
    scores = []
    enum_constraints = Gen.choicemap()
    for i in 1:num_dots
        if trace[(:isvisible, dot)]
            enum_constraints[(:x_pos, i)] = trace[(:x_pos, i)]
            enum_constraints[(:y_pos, i)] = trace[(:y_pos, i)]
            enum_constraints[(:offset_x, i)] = trace[(:offset_x, i)]
            enum_constraints[(:offset_y, i)] = trace[(:offset_y, i)]
        end
    end
    for eg in edge_truthtable
        for (eg_id, e) in enumerate(eg)
            if e == 1
                enum_constraints[(:edge, possible_edges[eg_id][1], possible_edges[eg_id][2])] = true
            else
                enum_constraints[(:edge, possible_edges[eg_id][1], possible_edges[eg_id][2])] = false
            end
        end
        
        for kc in kernel_combos
            for (dot, k) in enumerate(kc)
                enum_constraints[(:kernel_type, dot)] = k
                println(k)
            end
            # clears order from the original trace
            constraints_no_noise_order = [map_entry for map_entry in get_values_shallow(enum_constraints) if !(typeof(map_entry[1]) == Symbol)]
            constraints_no_params = [map_entry for map_entry in constraints_no_noise_order if !(map_entry[1][1] in [:amplitude, :variance, :covariance, :period, :lengthscale])]
            enum_constraints = Gen.choicemap(constraints_no_params...)
            kernel_assignments = [enum_constraints[(:kernel_type, i)] for i in 1:num_dots]
            collect_params_per_ktype = [Iterators.product([param_dict[ktype]..., param_dict[ktype]...]...) for ktype in kernel_assignments]
            all_param_permutations = Iterators.product(collect_params_per_ktype...)
            # this is totally correct. for random random, gives all combinations in form of ((10, 10), (10, 11))
            scene_scores = []
            for dp in all_dot_permutations(num_dots)
                enum_constraints[:order_choice] = dp
                for jc in jitter_combos
                    for (dot, j) in enumerate(jc)
                        enum_constraints[(:jitter, dot)] = j
                    end
                    for noise in param_dict[:noise]
                        enum_constraints[:noise] = noise
                        for param_permutation in all_param_permutations
                            param_assignments = hyper_permutation_to_assignment(kernel_assignments, param_permutation)
                            for p in param_assignments
                                enum_constraints[p.first] = p.second
                            end
                            (tr, w) = Gen.generate(generate_dotmotion, trace_args, enum_constraints)
                            append!(scene_scores, w)
                            if w > top_score
                                top_trace = tr
                                top_score = w
                                println("top score")
                                println(top_score)
                            end
                        end
                    end
                end
            end
            push!(scores, scene_scores)
        end
    end
    total_score = logsumexp(convert(Array{Float64, 1}, vcat(scores...)))
    println("TOTAL SCORE")
    score_scenegraph = [sum(map(x-> exp(x - total_score), sc)) for sc in scores]
    println(score_scenegraph)
    score_matrix = reshape(score_scenegraph, prod(collect(size(kernel_combos))), size(edge_truthtable)[1])
    plotvals = [score_matrix, kernel_combos, possible_edges, edge_truthtable]
    return plotvals, top_trace
end




function calculate_pairwise_distance(dotmotion_tuples)
    # each value in dotmotion_tuples is of form ((x1, y1), (x2, y2)) for each dot i to N.
    pairwise_distances = []
    for dt in dotmotion_tuples
        push!(pairwise_distances,
              [norm(coord2 .- coord1) for (i, coord1) in enumerate(dt) for (j, coord2) in enumerate(dt) if i < j])
    end
    return pairwise_distances
end



""" this is fine for when you have a trace, but if its a dot array, you have to specify the dot IDs first, 
and the offsets won't be observable in a true sense -- only the position of the observed dots """


function visualize_importance_sampling_results(all_importance_samples)
    top_samples, probabilities = top_imp_results(all_importance_samples)
    plot_inference_results(top_samples, probabilities)
    return all_importance_samples
end

function animate_imp_inference(all_importance_samples, groundtruth_trace)
    num_dots = groundtruth_trace[:num_dots]
    res = 700
    ts = get_args(groundtruth_trace)[1]
    inference_anim_fig = Figure(resolution=(num_dots*res, res), backgroundcolor=:white, outer_padding=0)
    dotmotion_axis =  [Axis(inference_anim_fig, showaxis = false, 
                            xgridvisible = false, 
                            ygridvisible = false, 
                            xticksvisible = false,
                            yticksvisible = false,
                            xticklabelsvisible = false,
                            yticklabelsvisible = false,
                            leftspinevisible= false,
                            rightspinevisible = false,
                            topspinevisible = false,
                            bottomspinevisible = false) for i in 1:num_dots]
    [inference_anim_fig[i, 1] = dm for (i, dm) in enumerate(dotmotion_axis)]
    display(inference_anim_fig)
    for dot in 1:num_dots
        lines!(dotmotion_axis[dot], groundtruth_trace[(:x_observable, dot)], groundtruth_trace[(:y_observable, dot)], color=:red)
        for trace in all_importance_samples
            m_graph = get_retval(trace)
            jitter = (trace[(:jitter, dot, :x)], trace[(:jitter, dot, :y)])
            noise = (trace[(:perceptual_noise, dot, :x)], trace[(:perceptual_noise, dot, :y)])
            start_x = trace[(:start_x, dot)]
            start_y = trace[(:start_y, dot)]
            x_timeseries = trace[(:x_timeseries, dot)]
            y_timeseries = trace[(:y_timeseries, dot)]
            pvel_x, pvel_y = get_parent_influence(m_graph, dot, ts)
            x_obs, y_obs = create_observable(x_timeseries, y_timeseries, jitter, noise, start_x, start_y, pvel_x, pvel_y)
            lines!(dotmotion_axis[dot], x_obs, y_obs, color=:gray)
            sleep(.3)
        end
    end

end

function imp_inference(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}},
                       observed_data)
    trace_choices = get_choices(trace)
    args = get_args(trace)
    observations = Gen.choicemap()
    all_samples = []
    all_graphs = []
    num_dots = trace[:num_dots]
    num_particles = 10000
    #    num_resamples = 30
    observations[:num_visible] = trace[:num_visible]
    for i in 1:observations[:num_visible]
        observations[(:start_x, i)] = trace[(:start_x, i)]
        observations[(:start_y, i)] = trace[(:start_y, i)]
        observations[(:x_observable, i)] = trace[(:x_observable, i)]
        observations[(:y_observable, i)] = trace[(:y_observable, i)]
    end
    [observations[constraint] = observed_data[constraint] for constraint in keys(observed_data)]
    (traces, weights, lml_est) = Gen.importance_sampling(generate_dot_scene, args, observations, dotnum_and_timeseries_proposal, (trace,), num_particles)
    
    # for i in 1:num_resamples
    #     (tr, w) = Gen.importance_resampling(generate_dot_scene, args, observations, dotnum_and_type_proposal, (trace,), num_particles)
    #    # (tr, w) = Gen.importance_resampling(generate_dot_scene, args, observations, num_particles)
    #     push!(all_samples, tr)
    # end
    
#    total_score = Gen.logsumexp(convert(Array{Float64, 1}, weights))
 #   trace_probabilities = [exp(w - total_score) for w in weights]
  #  println(sort(trace_probabilities)[1:200])
    all_samples_ordered = reverse(traces[sortperm(weights)])
#    all_weights_ordered = reverse(weights[sortperm(trace_probabilities)])
 #   println(all_weights_ordered[1:50])
    return all_samples_ordered
end


function top_imp_results(all_samples)
    # want the top graphs and probabilities
    all_graphs = []
    for g in [deepcopy(get_retval(tr)) for tr in all_samples]
        for i in 1:nv(g)
            for ky in keys(props(g, i))
                if !(ky in [:MType, :isvisible])
                    rem_prop!(g, i, ky)
                end
            end
        end
        push!(all_graphs, g)
    end
    top_graphs, probabilities = find_top_metagraphs(all_graphs, [])
    top_n = min(3, length(top_graphs))
    return top_graphs[1:top_n], probabilities[1:top_n]
end


function find_top_metagraphs(graphlist, unique_graphs)
    if isempty(graphlist)
        sorted_graphs = sort(unique_graphs, by= l -> length(l), rev=true)
        total_graphs = length(vcat(unique_graphs...))
        return [sg[1] for sg in sorted_graphs], [length(sg) / total_graphs for sg in sorted_graphs]
    else
        eq_to_element1 = map(x -> x == graphlist[1], graphlist)
        push!(unique_graphs, graphlist[eq_to_element1])
        find_top_metagraphs(graphlist[.!eq_to_element1], unique_graphs)
    end
end


# here, you'll have to construct a length top_graphs by 2 Figure.
# add a barplot onto the bottom row, and across the top add Axis objects that you
# can pass to visualize_scenegraph with the trace. 

function plot_inference_results(top_graphs, probabilities)
    scene, layout = layoutscene(resolution=(300, 300), backgroundcolor=RGBf0(0, 0, 0))
    white = RGBf0(255,255,255)
    black = RGBf0(0,0,0)
    gray = RGBf0(100, 100, 100)
    res = 500
    infres_fig = Figure(resolution=(3*res, 2*res), backgroundcolor=white, outer_padding=0)

    scenegraph_axes = [Axis(infres_fig, showaxis = false, 
                            xgridvisible = false, 
                            ygridvisible = false, 
                            xticksvisible = false,
                            yticksvisible = false,
                            xticklabelsvisible = false,
                            yticklabelsvisible = false,
                            backgroundcolor = white,
                            topspinevisible = false,
                            bottomspinevisible = false,
                            rightspinevisible = false,
                            leftspinevisible = false) for g in top_graphs]
    [ax.aspect = DataAspect() for ax in scenegraph_axes]
    infres_fig[1, 1:3] = scenegraph_axes
    vizgraphs = [visualize_scenegraph(g, ax) for (g, ax) in zip(top_graphs, scenegraph_axes)]
    bar_axis = Axis(infres_fig, backgroundcolor=white)
    infres_fig[2,:] = bar_axis
    barwidth = (.2 / 3) * length(probabilities)
    if length(probabilities) == 1
        bar_x = [.45]
        xaxlims = BBox(0, 1, 0, 1)
    elseif length(probabilities) == 2
        bar_x = [.4, 1.55]
        xaxlims = BBox(0, 2, 0, 1)
    elseif length(probabilities) == 3
        bar_x = [.35, 1.45, 2.55]
        xaxlims = BBox(0, 3, 0, 1)
    end
    bp = barplot!(bar_axis, bar_x, convert(Array{Float64, 1}, probabilities),
                  color=:gray,
                  backgroundcolor=:white, width=barwidth)
    bar_axis.ylabel = "Posterior Probability"
    limits!(bar_axis, xaxlims)
    screen = display(infres_fig)
    stop_anim = false
    on(events(infres_fig.scene).keyboardbuttons) do button
        if ispressed(button, Keyboard.enter)
            stop_anim = true
        end
    end
    query_enter() = stop_anim
    # will either stop when you press enter or when 200 seconds have passed. 
    timedwait(query_enter, 200.0)
    # uncomment if you want to block until the window is closed
    #  wait(screen)
end

# Params are of form 
#((dot1x, dot1y), (dot2x, dot2y))
#((10, 10), (10, 10))
#((amp1x,l1x,period1x,amp1y,l1y,period1y), (1,2,3,4,5,6))


function hyper_permutation_to_assignment(kernels, params)
    pdict = Dict()
    dot_dim_combos = [(dot, dim) for dot in 1:length(kernels) for dim in 1:2]
    for (dot, dim)  in dot_dim_combos
        if kernels[dot] == Periodic
            pdict[(:amplitude, dot, dim)] = params[dot][dim^2]
            pdict[(:lengthscale, dot, dim)] = params[dot][1+dim^2]
            pdict[(:period, dot, dim)] = params[dot][2+dim^2]
        elseif kernels[dot] == RandomWalk
            pdict[(:variance, dot, dim)] = params[dot][dim]
        elseif kernels[dot] == Linear
            pdict[(:covariance, dot, dim)] = params[dot][dim]
        end
    end
    return pdict
end


function filter_hyperparams(choices, c_graph, gt_choices, gt_graph, n, d)
    if c_graph == gt_graph && all([choices[(:kernel_type, i)] == gt_choices[(:kernel_type, i)] for i in 1:nv(gt_graph)])
        if choices[(:kernel_type, n)] == RandomWalk
            param = (:variance, choices[(:variance, n, d)])
        elseif choices[(:kernel_type, n)] == Periodic
            param =  [(:amplitude, choices[(:amplitude, n, d)]),
                      (:lengthscale, choices[(:lengthscale, n, d)]),
                      (:period, choices[(:period, n, d)])]
        elseif choices[(:kernel_type, n)] == Linear
            param = (:covariance, choices[(:covariance, n, d)])
        end
    else
        param = ()
        println("wrong graph")
        println(c_graph == gt_graph)
        println(c_graph)
        println(gt_graph)
        println(all([choices[(:kernel_type, i)] == gt_choices[(:kernel_type, i)] for i in 1:nv(gt_graph)]))
    end
    return param
        
end    


# only want hyper params for when edge AND kernel inference is correct.     
function hyperparameter_inference(importance_samples, groundtruth_trace)
    choicemaps = [get_choices(trace) for trace in importance_samples]
    importance_sample_graphs = [get_retval(trace)[1].graph for trace in importance_samples]
    gt_choicemap = get_choices(groundtruth_trace)
    gt_graph = get_retval(groundtruth_trace)[1].graph
    num_dots = nv(get_retval(groundtruth_trace)[1])
    num_dims = 2
    hyper_matrix = Dict()
    gt_hypermatrix = Dict()
    hyper_matrix_MAP = Dict()
    gt_hyper_comparison = []
    gt_hypermatrix[:noise] = gt_choicemap[:noise]
    hyper_matrix[:noise] = [choices[:noise] for choices in choicemaps]
    hyper_matrix_MAP[:noise] = findmax(countmap(hyper_matrix[:noise]))[2]                            
    for n in 1:num_dots
        for d in 1:num_dims
            hyper_matrix[(n, d)] = Any[]
            gt_hypermatrix[(n, d)] = filter_hyperparams(gt_choicemap, gt_graph, gt_choicemap, gt_graph, n, d)
            for (choices, c_graph) in zip(choicemaps, importance_sample_graphs)
                push!(hyper_matrix[(n, d)], filter_hyperparams(choices, c_graph, gt_choicemap, gt_graph, n, d))
            end
            hyper_matrix_MAP[(n, d)] = findmax(countmap(filter(x -> x != (), hyper_matrix[(n, d)])))[2]
        end                
    end
                            
    for key in keys(hyper_matrix_MAP)
        hyperparam = hyper_matrix_MAP[key]
        gtparam = gt_hypermatrix[key]
        if key == :noise
            push!(gt_hyper_comparison, (:noise, hyperparam-gtparam))
        else
        # filters for the scene graph inference being incorrect...no this doesn't work. 

            if isa(hyperparam, Array)
                push!(gt_hyper_comparison, (:amplitude, hyperparam[1][2] - gtparam[1][2]))
                push!(gt_hyper_comparison, (:lengthscale, hyperparam[2][2] - gtparam[2][2]))
                push!(gt_hyper_comparison, (:period, hyperparam[3][2] - gtparam[3][2]))
            else
                if hyperparam[1] == :variance
                    push!(gt_hyper_comparison, (:variance, hyperparam[2] - gtparam[2]))
                elseif hyperparam[1] == :covariance
                    push!(gt_hyper_comparison, (:covariance, hyperparam[2] - gtparam[2]))
                end
            end
        end
    end
    return hyper_matrix_MAP, gt_hypermatrix, gt_hyper_comparison
end

function aggregate_hyperparam_inference(hyperlist)
    hps = [:amplitude, :period, :lengthscale, :noise, :variance, :covariance]
    hp_dict = Dict()
    for hp in hps
        try
            hp_dict[hp] = [abs(v[2]) for v in hyperlist if v[1] == hp]
        catch
        end
    end
    return hp_dict
end    


function run_silent_inftest(num_iters::Int, num_dots::Int)
    traces = []
    for i in 1:num_iters
        trace, args = dotsample(num_dots);
        push!(traces, trace)
    end
    all_resamples = evaluate_inference_accuracy(traces)
    inference_hps, gt_hps, gt_hp_comparison = hyperparameter_inference(all_resamples[1], traces[1])
end    


function evaluate_inference_accuracy(traces)
    correct_counter = zeros(4)
    filtered_resamples_per_trace = []
    for t in traces
        num_dots = nv(get_retval(t)[1])
        resamples, e, v = imp_inference(t)
        filtered_resamples = []
        motion_tree = get_retval(t)[1]
        mp_edge = findmax(countmap(e))[2]
        mp_velocity = findmax(countmap(v))[2]
        (scoremat, kernels, p_edges, edge_tt), top_bayes_trace = bayesian_observer(t)
        max_score = findmax(scoremat)[2]
        max_enum_vel = kernels[max_score[1]]
        max_enum_edge = edge_tt[max_score[2]]
        edge_truth = [convert(Int64, has_edge(motion_tree, d1, d2))
                      for d1 in 1:num_dots for d2 in 1:num_dots if d1 != d2]
        velocity_truth = [t[(:kernel_type, d)] for d in 1:num_dots]
        # keep traces where with the max estimate scene graph 
        for rs in resamples
            edges = [rs[(:edge, j, k)] for j in 1:num_dots for k in 1:num_dots if j!=k]
            vel_types = [rs[(:kernel_type, j)] for j in 1:num_dots]
            if edges == mp_edge && vel_types == mp_velocity
                push!(filtered_resamples, rs)
            end
        end
        push!(filtered_resamples_per_trace, filtered_resamples)
        if Tuple(edge_truth) == max_enum_edge
            correct_counter[1] += 1
        end
        if Tuple(velocity_truth) == max_enum_vel
            correct_counter[2] += 1
        end
        if edge_truth == mp_edge
            correct_counter[3] += 1
        end
        if velocity_truth == mp_velocity
            correct_counter[4] += 1
        end
    end
    barplot(correct_counter / length(traces))
    return filtered_resamples_per_trace
end        



# Plotting function will now take the top 3 traces raw as well as their probabilities. 


""" These functions are for visualizing scenegraphs and animating dotmotion using Makie """


function trace_to_coords(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    visible_dots = [dot for dot in 1:trace[:num_dots] if trace[(:isvisible, dot)]]
    raw_dotmotion = fill(zeros(2), trace[:num_dots], length(trace[(:x_observable, 1)]))
    visible_dotmotion = fill(zeros(2), length(visible_dots), length(trace[(:x_observable, 1)]))
    if isempty(visible_dots)
        return [], [], [], []
    end
    vis_dot_index = 1
    for dot in 1:trace[:num_dots]
        raw_dotmotion[dot, :] = [[x, y] for (x, y) in zip(trace[(:x_observable, dot)], trace[(:y_observable, dot)])]
        if dot in visible_dots
            visible_dotmotion[vis_dot_index, :] = raw_dotmotion[dot,:]
            vis_dot_index += 1
        end
    end
    visible_dotmotion_tuples = [[Tuple(visible_dotmotion[i, j]) for i in 1:length(visible_dots)] for j in 1:size(visible_dotmotion)[2]]
    raw_dot_tuples = [[Tuple(raw_dotmotion[i, j]) for i in 1:trace[:num_dots]] for j in 1:size(raw_dotmotion)[2]]
    return visible_dotmotion_tuples, raw_dotmotion, raw_dot_tuples
end


# write your own version of the graph plotter. use arrow and scatter primitives in makie.
# just have to write a clever algorithm for placement of each dot.

# recursive function here takes an array of xcoords, ycoords, paths, and the graph
# init with zero arrays of length nv for x and y coords. n_iters too.
function xy_node_positions(paths::Array{Array, 1},
                           xc::Array{Int64, 1},
                           yc::Array{Int64, 1},
                           n_iters::Int,
                           motion_tree::MetaDiGraph{Int64, Float64},
                           longest_path::Int)
    if isempty(paths)
        xcoords = convert(Array{Float64, 1}, xc)
        ycoords = convert(Array{Float64, 1}, yc)

        # this is a fix for the location of multi input or output nodes 
        for v in 1:nv(motion_tree)
            inn = inneighbors(motion_tree, v)
            outn = outneighbors(motion_tree, v)
            if length(inn) > 1
                #     xcoords[v] = mean([xcoords[n] for n in inn])
                ycoords[v] -= (1 - length(inn))
            end
            if length(outn) > 1
         #       xcoords[v] = mean([xcoords[n] for n in outn])

            end
         end
         return xcoords, ycoords
    else    
        path = first(paths)
        # each path has its own x coord. first path goes to x = 1, and unassigned will have xcoord = 0 
        [xc[dot] == 0 ? xc[dot] = n_iters : xc[dot] = xc[dot] for dot in path]
        # reachable_to counts how many dots are connected in the path stemming from the current dot.
        [yc[dot] = longest_path - length(reachable_to(motion_tree.graph, dot)) for dot in path]
        xy_node_positions(paths[2:end], xc, yc, n_iters+1, motion_tree, longest_path)
    end
end


function node_color(mtype::String, dot_jitter_or_noise::Symbol)
    skyblue = RGBf0(154, 203, 255) / 255
    darkcyan = RGBf0(0, 170, 170) / 255
    lightgreen = RGBf0(144, 238, 144) / 255
    yellow = RGBf0(255, 240, 120) / 255
    red = RGBf0(220, 100, 100) / 255
    magenta = RGBf0(255,128,255) / 255
    dark_green = RGBf0(34,139,34) / 255
    dark_blue = RGBf0(0, 60, 120) / 255
    dark_red = RGBf0(120, 60, 0) / 255
    gray = RGBf0(130, 130, 130) / 255
    if mtype == "Linear"
        nodecolor = lightgreen
    elseif mtype == "Periodic"
        nodecolor = skyblue
    elseif mtype == "SquaredExponential"
        nodecolor = red
    elseif mtype == "LinearPeriodic"
        nodecolor = darkcyan
    elseif mtype == "LinearSquaredExponential"
        nodecolor = yellow
    elseif mtype == "LinearLinear"
        nodecolor = dark_green
    elseif mtype == "PeriodicSquaredExponential"
        nodecolor = magenta
    elseif mtype == "PeriodicPeriodic"
        nodecolor = dark_blue
    elseif mtype == "SquaredExponentialSquaredExponential"
        nodecolor = dark_red
    else
        nodecolor = :white
    end
    if dot_jitter_or_noise == :jitter
        nodecolor = (nodecolor, .6)
    elseif dot_jitter_or_noise == :perceptual_noise
        nodecolor = (nodecolor, .3)
    else
        nodecolor = (nodecolor, 1.0)
    end
    return nodecolor
end    

# here add all labels and hyperparams. make this a makie layout.
# there are only two ways to do this now. either make an independent scene then a new LScene in render_dotmotion with its
# .scene field equal to the returned scene, or add an axis into visualize scenegraph, and manipulate it inside the function.
# both work, but then visualize scenegraph becomes a manipulator of LAxis objects and unable to render itself. 

function visualize_scenegraph(motion_tree::MetaDiGraph{Int64, Float64},
                              sg_axis::Axis)
    res = 1400
    paths = all_paths(motion_tree)
    # by the end of this loop have all connected and all unconnected paths
    for v in 1:nv(motion_tree)
        v_in_path = [v in p ? true : false for p in paths]
        if !any(v_in_path)
            push!(paths, [v])
        end
    end
    longest_path = maximum(map(length, paths))
    num_paths = length(paths)
    xbounds = num_paths + 1
    ybounds = longest_path + 1
    node_xs, node_ys = xy_node_positions(paths, zeros(Int, nv(motion_tree)), zeros(Int, nv(motion_tree)), 1, motion_tree, longest_path)
    for e in edges(motion_tree)
        arrows!(sg_axis, [node_xs[e.src]], [node_ys[e.src]],
                .8 .* [node_xs[e.dst]-node_xs[e.src]], .8 .* [node_ys[e.dst]-node_ys[e.src]],
                arrowcolor=:gray, linecolor=:gray, arrowsize=.1)
    end
    for v in 1:nv(motion_tree)
        mtype = props(motion_tree, v)[:MType]
        nodecolor = node_color(mtype, :dot)
        scatter!(sg_axis, [(node_xs[v], node_ys[v])], markersize=50, color=nodecolor),
        text!(sg_axis, string(v), position=(node_xs[v], node_ys[v]), align= (:center, :center),
              textsize=.1, color= props(motion_tree, v)[:isvisible] ? :black : :white, overdraw=true)
# Commented Regions Show the Noise and Jitter on the Graph         
#        text!(sg_axis, string("s_", v, " = ", choices[(:jitter_magnitude, v)]), 
 #             position=(node_xs[v] + .32, node_ys[v] + -.13), align= (:center, :center),
  #            textsize=.08, color=:gray, overdraw=true)
     #   plot!(choices[(:x_timeseries, v)], color=:blue)
     #   plot!(choices[(:y_timeseries, v)], color=:red)

    end
  #  text!(sg_axis, string("eps = ", choices[:perceptual_noise_magnitude]), 
   #       position=(node_xs[1] + .1, node_ys[1] + .4), align= (:center, :center),
    #      textsize=.1, color=:gray, overdraw=true)
    xlims!(sg_axis, 0, xbounds)
    ylims!(sg_axis, 0, ybounds)
    sg_axis.aspect = DataAspect()
    return sg_axis
end


function find_top_n_props(n::Int,
                          score_matrix,
                          max_inds)
    mi = findmax(mappedarray(x-> isfinite(x) ? x : 0.0, score_matrix))
    if n == 0 || mi[1] == 0
        return max_inds
    else
        mi_coord = mi[2].I
        push!(max_inds, mi)
        sm_copy = copy(score_matrix)
        sm_copy[mi_coord[1], mi_coord[2]] = 0.0
        find_top_n_props(n-1, sm_copy, max_inds)
    end
end    


function render_dotmotion(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}, show_scenegraph::Bool, show_timeseries::Bool)
  #  motion_tree = get_retval(trace)
    bounds = 25
    res = 1000
    outer_padding = 0
    dotmotion, raw_dotmotion, raw_dot_tuples = trace_to_coords(trace)
    number_timepoints = length(trace[(:x_timeseries, 1)])
    if isempty(dotmotion)
        println("NO VISIBLE DOTS")
        return [], []
    end
    visible_dots = [dot for dot in 1:trace[:num_dots] if trace[(:isvisible, dot)]]
    pairwise_distances = calculate_pairwise_distance(raw_dotmotion)
    stationary_duration = 50
    stationary_coords = [dotmotion[1] for i in 1:stationary_duration]
    raw_stationary_coords = [raw_dot_tuples[1] for i in 1:stationary_duration]
    time_node = Node(1);
    f(t, coords) = coords[t]
    f_color(t) = t < stationary_duration ? :white : :black
    f_textsize(t) = t < stationary_duration ? 2 : 0
    f_timeseries(t, coords) = t < stationary_duration ? [coords[1]] : coords[1:t-stationary_duration+1]
    n_rows = 3
    n_cols = 2
    white = RGBf0(255,255,255)
    black = RGBf0(0,0,0)
    if show_timeseries
        dotmotion_fig = Figure(resolution=(2*res, 2*res), backgroundcolor=white, outer_padding=0)

        """ Uncomment if you want to plot static scenegraph in Panel 2"""
        # scenegraph_axis = Axis(dotmotion_fig, showaxis = false, 
        #                        xgridvisible = false, 
        #                        ygridvisible = false, 
        #                        xticksvisible = false,
        #                        yticksvisible = false,
        #                        xticklabelsvisible = false,
        #                        yticklabelsvisible = false,
        #                        backgroundcolor = white,
        #                        title = "Scene Graph")
        # scenegraph_axis.aspect = DataAspect()
        # dotmotion_fig[1, 2] = visualize_scenegraph(motion_tree, get_choices(trace), scenegraph_axis)
        dotmotion_fig[1, 3] = Legend(dotmotion_fig,
                                     [MarkerElement(color=:orange, marker=:circle, strokecolor=:black),
                                      MarkerElement(color=:skyblue, marker=:circle, strokecolor=:black),
                                      MarkerElement(color=:lightgreen, marker=:circle, strokecolor=:black)],
                                     ["SqExp", "Periodic", "Linear"], orientation=:vertical)
        offset_axis = [Axis(dotmotion_fig,
                            backgroundcolor = white, title=string("Indep Component Dot ", i)) for i in 1:trace[:num_dots]]
        timeseries_axis = [Axis(dotmotion_fig,
                                backgroundcolor = white, title=string("Final Timeseries Dot ", i)) for i in 1:trace[:num_dots]]
        ts_subscene = dotmotion_fig[2:3, :]
        for i in 1:trace[:num_dots]
            ts_subscene[i, 1] = offset_axis[i]
            ts_subscene[i, 2] = timeseries_axis[i]
            cl = []
            motion_type = mtype_extractor(trace[(:cf_tree, i)][1])
            cl = [node_color(motion_type, :dot)[1]*.5, node_color(motion_type, :dot)[1]]
            ofs_x = lines!(offset_axis[i], lift(t -> f_timeseries(t, trace[(:x_timeseries, i)] .+= (trace[(:start_x, i)] - trace[(:x_timeseries, i)][1])), time_node), color=cl[1])
            ofs_y = lines!(offset_axis[i], lift(t -> f_timeseries(t, trace[(:y_timeseries, i)] .+= (trace[(:start_y, i)] - trace[(:y_timeseries, i)][1])), time_node), color=cl[2])
#            axislegend(offset_axis[i], [ofs_x, ofs_y], [string([string(h, "=", trace[(h, i, 1)], " ") for h in hp]...),
 #                                                       string([string(h, "=", trace[(h, i, 2)], " ") for h in hp]...)], position=:lt)
            ts_x = lines!(timeseries_axis[i], lift(t -> f_timeseries(t, trace[(:x_observable, i)]), time_node), color=cl[1])
            ts_y = lines!(timeseries_axis[i], lift(t -> f_timeseries(t, trace[(:y_observable, i)]), time_node), color=cl[2])
            [xlims!(t_ax, 0, number_timepoints) for t_ax in [offset_axis; timeseries_axis]]
            [ylims!(t_ax, -bounds, bounds) for t_ax in [offset_axis; timeseries_axis]]
        end
    else
        dotmotion_fig = Figure(resolution=(2*res, res), backgroundcolor=black, outer_padding=0)
    end

    """ Panel 1 """ 
    
    motion_axis = dotmotion_fig[1, 1] = Axis(dotmotion_fig, showaxis = false, 
                                             xgridvisible = false, 
                                             ygridvisible = false, 
                                             xticksvisible = false,
                                             yticksvisible = false,
                                             xticklabelsvisible = false,
                                             yticklabelsvisible = false,
                                             leftspinevisible= false,
                                             rightspinevisible = false,
                                             topspinevisible = false,
                                             bottomspinevisible = false, 
                                             backgroundcolor = black)
    motion_axis.aspect = DataAspect()
    for dot in visible_dots
        textloc = (trace[(:x_observable, dot)][1], trace[(:y_observable, dot)][1])
        text!(motion_axis, string(dot), position = (textloc[1], textloc[2] + 1), color=white, 
              textsize=lift(t -> f_textsize(t), time_node))
    end
    scatter!(motion_axis, lift(t -> f(t, [stationary_coords; dotmotion]), time_node), markersize=14px, color=RGBf0(255, 255, 255))
    xlims!(motion_axis, (-bounds, bounds))
    ylims!(motion_axis, (-bounds, bounds))


    """ Panel 2 """ 
    if show_scenegraph
        scenegraph_animation_axis = dotmotion_fig[1, 2] = Axis(dotmotion_fig, showaxis = false, 
                                                               xgridvisible = false, 
                                                               ygridvisible = false, 
                                                               xticksvisible = false,
                                                               yticksvisible = false,
                                                               xticklabelsvisible = false,
                                                               yticklabelsvisible = false,
                                                               leftspinevisible= false,
                                                               rightspinevisible = false,
                                                               topspinevisible = false,
                                                               bottomspinevisible = false, 
                                                               backgroundcolor = black)
        scenegraph_animation_axis.aspect = DataAspect()
        visdotind = 0
        invisdotind = 0
        marker_size = 1
        for dot in 1:trace[:num_dots]
            node_color_input = mtype_extractor(trace[(:cf_tree, dot)][1])
            jitter_std = sqrt(trace[(:jitter_magnitude, dot)])
            perceptual_noise_std = sqrt(trace[(:perceptual_noise_magnitude, dot)])
            scatter!(scenegraph_animation_axis, lift(t -> f(t, [map(λ -> [λ[dot]], raw_stationary_coords); map(λ -> [λ[dot]], raw_dot_tuples)]), time_node), markersize=marker_size + 2*jitter_std,
                     markerspace=SceneSpace, color=node_color(node_color_input, :jitter), overdraw=true)
            scatter!(scenegraph_animation_axis, lift(t -> f(t, [map(λ -> [λ[dot]], raw_stationary_coords); map(λ -> [λ[dot]], raw_dot_tuples)]), time_node), markersize=marker_size + 2*perceptual_noise_std,
                     markerspace=SceneSpace, color=node_color(node_color_input, :perceptual_noise), overdraw=true)
            textloc = (trace[(:x_observable, dot)][1]+.5, trace[(:y_observable, dot)][1])
            text!(scenegraph_animation_axis, string(dot), position = (textloc[1], textloc[2] + 1), color=white, textsize=lift(t-> f_textsize(t), time_node), overdraw=false)
            if dot in visible_dots
                scatter!(scenegraph_animation_axis, lift(t -> f(t, [map(λ -> [λ[dot]], raw_stationary_coords); map(λ -> [λ[dot]], raw_dot_tuples)]), time_node), markersize=marker_size,
                         markerspace=SceneSpace, color=node_color(node_color_input, :dot), overdraw=true)
                """ Jitter and noise each have their own circles """
            else
                scatter!(scenegraph_animation_axis,
                         lift(t -> f(t, [map(λ -> [λ[dot]], raw_stationary_coords);
                                         map(λ -> [λ[dot]], raw_dot_tuples)]), time_node), markersize=marker_size*10, color=RGBf0(0,0,0),
                         stroke_color=node_color(node_color_input, :dot)[1], strokewidth=.1)
            end
        end

        for e1 in 1:trace[:num_dots]
            for e2 in 1:trace[:num_dots]
                if e1 == e2
                    continue
                end
                if trace[(:edge, e1, e2)]
                    x_source = [raw_dotmotion[e1,:][1][1] .* ones(stationary_duration) ; [x[1] for x in raw_dotmotion[e1,:]]]
                    y_source = [raw_dotmotion[e1,:][1][2] .* ones(stationary_duration) ; [y[2] for y in raw_dotmotion[e1,:]]]
                    x_vec = [raw_dotmotion[e2,:][1][1] .* ones(stationary_duration) ; [x[1] for x in raw_dotmotion[e2,:]]] .- x_source
                    y_vec = [raw_dotmotion[e2,:][1][2] .* ones(stationary_duration) ; [y[2] for y in raw_dotmotion[e2,:]]] .- y_source
                    arrows!(scenegraph_animation_axis,
                            lift(t -> [f(t, x_source)], time_node),
                            lift(t -> [f(t, y_source)], time_node),
                            lift(t -> [.9*f(t, x_vec)], time_node),
                            lift(t -> [.9*f(t, y_vec)], time_node),
                            arrowcolor=:white, linecolor=:white, arrowsize=1)
                end
            end
        end
        xlims!(scenegraph_animation_axis, (-bounds, bounds))
        ylims!(scenegraph_animation_axis, (-bounds, bounds))
    end
    # Uncomment if you want to visualize scenegraph side by side with stimulus
    screen = display(dotmotion_fig)
    #    record(gt_scene, "stimulus.mp4", 1:size(dotmotion)[1]; framerate=60) do i
    #    for i in 1:size(dotmotion)[1]
    i = 0
    num_repeats = 0
    #    isopen(scene))
    stop_anim = false
    on(events(dotmotion_fig.scene).keyboardbuttons) do button
        if ispressed(button, Keyboard.enter)
            stop_anim = true
        end
    end
    while(!stop_anim)
        i += 1
        if i == size([stationary_coords; dotmotion])[1]
            sleep(2)
            i = 1
            num_repeats += 1
        end
        time_node[] = i
        sleep(1/framerate)
    end
#    return pairwise_distances, num_repeats
    return dotmotion
end


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

    

"""Linear kernel"""
struct Linear <: PrimitiveKernel
    c::Float64
    σ::Float64
end

eval_cov(node::Linear, t1, t2) = node.σ^2 * (t1 - node.c) * (t2 - node.c)

function eval_cov_mat(node::Linear, ts::Array{Float64})
    ts_minus_param = ts .- node.c
    node.σ^2 .* (ts_minus_param * ts_minus_param')
end


# note that in the gaussian process lit the subtraction here is often the norm

"""Squared exponential kernel"""
struct SquaredExponential <: PrimitiveKernel
    amplitude::Float64
    length_scale::Float64
end

eval_cov(node::SquaredExponential, t1, t2) =
    node.amplitude * exp(-0.5 * (t1 - t2) * (t1 - t2) / (node.length_scale ^ 2))

function eval_cov_mat(node::SquaredExponential, ts::Array{Float64})
    diff = ts .- ts'
    node.amplitude .* exp.(-0.5 .* diff .* diff ./ (node.length_scale ^ 2))
end

"""Periodic kernel"""
struct Periodic <: PrimitiveKernel
    amplitude::Float64
    lengthscale::Float64
    period::Float64
end


function eval_cov(node::Periodic, t1, t2)
    (node.amplitude ^ 2) * exp(
        (-2/node.lengthscale^2) * sin(pi*abs(t1-t2)/node.period)^2) 
end

function eval_cov_mat(node::Periodic, ts::Array{Float64})
    abs_diff = abs.(ts .-ts')
    (node.amplitude ^ 2) .* exp.((-2/node.lengthscale^2) .* sin.(pi*abs_diff./node.period).^2) 
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
# note this will come in handy when estimating the parameters of the function
# currently using deterministic params. 

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


function mtype_extractor(node::Kernel)
    if typeof(node) != Plus
        return string(typeof(node))
    else
        return string(
        sort([mtype_extractor(node.left), mtype_extractor(node.right)])...)
    end
end



""" PARAMS FOR GAUSSIAN PROCESS AND KERNEL CHOICES """

framerate = 25
time_duration = 6
num_position_points = time_duration * framerate
timepoints = convert(Array{Float64}, range(1, stop=time_duration, length=num_position_points))

# can't interpolate in the position based model -- then you get linear motion inside the periodic and random motion. 

# param_dict = Dict(Periodic => [[3], [.5, 1], [1]],
#                   Linear => [[.5], [6, 12]],
#                   SquaredExponential => [[2, 10], [.05]],
#                   :jitter_magnitude => [0, .01, .1, 1], 
#                   :perceptual_noise_magnitude => [0, .01, .1, 1])

# periodic is amp, length, period

# HAVE TO ASSESS THE MEANING OF THESE PARAMS TO MAKE GOOD PROPOSALS

# .5 went exactly every 15 frames
# 1.25 every ~40 frames (probably 38)
#1.5 every 45 frames. nice.
# amplitude basically translates directly. normal on the peak to peak is probably a good choice. 
# lengthscale is on the same scale as period. .5 is probably 15 frames worth of covariance. 


function test_param_set(mn_probs)
    cf_x, cv_y = covfunc_prior(mn_probs, 1, 0)
    covmat_x = compute_cov_matrix_vectorized(cf_x, ϵ, timepoints)
    println(cf_x)
    draws = [mvnormal(zeros(length(timepoints)), covmat_x) for i in 1:30]
    fig, ax = lines(draws[1])
    for d in draws[2:end]
        lines!(ax, d)
    end
    display(fig)
    return covmat_x
end

param_dict = Dict(Periodic => [collect(2:10), collect(.5:.5:2), collect(.4:.2:3)],
                  Linear => [[0], collect(0:.5:20)],
                  SquaredExponential => [collect(1:20), collect(.05:.05:.5)],
                  :jitter_magnitude => [0],    # previously [.01, .1, 1]
                  :perceptual_noise_magnitude => [0.0])

ϵ = 1e-6
# need a tiny bit of noise in mvnormal else factorization errors. this is also noted in Gaussian Process book. 
# if you use offsets from previous dots instead of biases to timeseries, only the
# inheriting dots will ever be displaced from the center. 

#kernel_types = [Periodic, SquaredExponential, Linear]
kernel_types = [Periodic, SquaredExponential, Linear, Plus]
@dist choose_kernel_type() = kernel_types[categorical([1/3, 1/3, 1/3])]
@dist choose_kernel_type(mn_probs) = kernel_types[categorical(mn_probs)]



@gen function covariance_prior(kernel_type, dot, dim)
    if kernel_type == Periodic
        lengthscale = {(:lengthscale, dot, dim)} ~ multinomial(param_dict[Periodic][2])
        period = {(:period, dot, dim)} ~ multinomial(param_dict[Periodic][3])
        #     amplitude = {(:amplitude, dot, dim)} ~ multinomial([50/period])
        amplitude = {(:amplitude, dot, dim)} ~ multinomial(param_dict[Periodic][1])
        kernel_args = [amplitude, lengthscale, period]
    elseif kernel_type == Linear
        lengthscale =  {(:offset, dot , dim)} ~ multinomial(param_dict[Linear][1])  
        variance = {(:variance, dot, dim)} ~ multinomial(param_dict[Linear][2])
        kernel_args = [lengthscale, variance]
    elseif kernel_type == RandomWalk
        kernel_args = [{(:variance, dot, dim)} ~ multinomial(param_dict[RandomWalk][1])]
    elseif kernel_type == SquaredExponential
        kernel_args = [{(:amplitude, dot, dim)} ~ multinomial(param_dict[SquaredExponential][1]),
                       {(:lengthscale, dot , dim)} ~ multinomial(param_dict[SquaredExponential][2])]        
    end
    return kernel_type(kernel_args...)
end



# in covfunc prior, sample two param sets for every kernel -- for x and y.
# each node will now carry a param for x and y. in our paradigm, a dot moving
# with uniform x and periodic y is simply a periodic dot inheriting motion of a uniform dot.
# hyperparam scaling will minimize x or y contribution in this case.
# a Plus is an internal node, and everything else is a Leaf.
#


# Eliminate SqExp from the possibilities in the proposal.
# make the pluscounter 1 to start in the proposal. 

@gen function covfunc_prior(mn_probs, dot, pluscounter)
    # if pluscounter == 1
    #     mn_probs[1] += mn_probs[4] / 2
    #     mn_probs[3] += mn_probs[4] / 2
    #     mn_probs[4] = 0
    # end
    kernel_type = { :kernel_type } ~ choose_kernel_type(mn_probs)
    if in(kernel_type, [Plus, Times])
        pluscounter += 1
        left = { :left } ~ covfunc_prior([1, 0, 0, 0], dot, pluscounter)
        right = { :right } ~ covfunc_prior([0, 0, 1, 0], dot, pluscounter)
        return kernel_type(left[1], right[1]), kernel_type(left[2], right[2])
    end
    if kernel_type == Periodic
        amplitude =[{(:amplitude, i)} ~ multinomial(param_dict[Periodic][1]) for i in [:x, :y]]
        lengthscale = [{(:lengthscale, i)} ~ multinomial(param_dict[Periodic][2]) for i in [:x, :y]]
        period = [{(:period, i)} ~ multinomial(param_dict[Periodic][3]) for i in [:x, :y]]
        kernel_args_x, kernel_args_y = zip(amplitude, lengthscale, period)
    elseif kernel_type == Linear
        lengthscale =  [{(:offset, i)} ~ multinomial(param_dict[Linear][1]) for i in [:x, :y]]
        variance = [{(:variance, i)} ~ multinomial(param_dict[Linear][2]) for i in [:x, :y]]
        kernel_args_x, kernel_args_y = zip(lengthscale, variance)
    elseif kernel_type == SquaredExponential
        amplitude = [{(:amplitude, i)} ~ multinomial(param_dict[SquaredExponential][1]) for i in [:x, :y]]
        lengthscale = [{(:lengthscale, i)} ~ multinomial(param_dict[SquaredExponential][2]) for i in [:x, :y]]
        kernel_args_x, kernel_args_y = zip(amplitude, lengthscale)
    end
    return kernel_type(kernel_args_x...), kernel_type(kernel_args_y...)
end    

@dist gamma_bounded_below(shape, scale, bound) = gamma(shape, scale) + bound

@dist poisson_bounded_below(rate, bound) = poisson(rate) + bound

@dist geometric_bounded_below(prob, bound) = geometric(prob) + bound

@dist multinomial(possibilities) = possibilities[uniform_discrete(1, length(possibilities))]


macro datatype(str); :($(Symbol(str))); end

"""
Length-`k` "random partial permutation" of the integers 1 through `n`
inclusive.
That is, a uniformly random sequence of `k` distinct integers between 1 and `n`
inclusive.
"""
struct RandomPartialPermutation <: Gen.Distribution{Vector{Int64}} end
const randomPartialPermutation = RandomPartialPermutation()
(::RandomPartialPermutation)(n, k) = Gen.random(randomPartialPermutation, n, k)

# Note: if `n` is huge and `k` is small, an implementation that doesn't
# materialize `collect(1:n)` would be preferable.
function Gen.random(::RandomPartialPermutation, n::Integer, k::Integer)
  return StatsBase.sample(1:n, k; replace=false)
end

function Gen.logpdf(::RandomPartialPermutation, pperm::Vector{Int64},
                    n::Integer, k::Integer)
  @assert length(pperm) == k
  @assert k <= n
  # Number of possible values for `pperm` is n! * (n-k)!
  return -(loggamma(n + 1) - loggamma(n - k + 1))
end

Gen.has_output_grad(::RandomPartialPermutation) = false
Gen.has_argument_grads(::RandomPartialPermutation) = (false, false)
Gen.logpdf_grad(::RandomPartialPermutation, n, k) = (nothing, nothing, nothing)
# Note that periodic motion can't accelerate linearly. This is clearly a primitive.
# Eventually want to use a 0 to 50 scale w a mid prior on position. then you can
# wrap the coords when the dot leaves so it comes back on the other side.
# Could either do that or just make the screen huge. It's moving by pixels so you can just expand the
# screen to capture the dynamics, and increase the bounds with it. 

# thinking a custom proposal could be zero mean on the mvnormal draw. kick anything out that doesn't have it. 



""" These functions are for creating, administering, and analyzing human data """


function assign_task_filename(num_dots::Int, directory::String)
    task_filenames = readdir(directory)
    taskfile_index = 1
    while(true)
        file_id = string("tasktrace", num_dots, taskfile_index, ".bson")
        if !(file_id in task_filenames)
            return file_id
        else
            taskfile_index += 1
        end
    end
end


function find_samples_for_task(num_dots::Int, add_to_current_samples::Bool)
    human_task_directory = "/Users/nightcrawler2/humantest"
    if !add_to_current_samples
        for filename in readdir(human_task_directory)
            try
                if filename[1:length("tasktrace")+1] == string("tasktrace", num_dots)
                    rm(string(human_task_directory, "/", filename))
                end
            catch
            end
        end
    end
    while(true)
        trace, args = dotsample(num_dots)
        render_dotmotion(trace, true)
        println("keep trace?")
        ans1 = readline()
        if ans1 != "y"
            continue
        end
        inf_results = animate_inference(trace)
        analyze_and_plot_inference(inf_results[1:4]...)
        println("keep trace?")
        ans2 = readline()
        if ans2 == "y"
            trace_file_id = assign_task_filename(num_dots, human_task_directory)
            trace_args, trace_choices = get_args(trace), get_choices(trace)
            @save string(human_task_directory,"/", trace_file_id) trace_args trace_choices
            println("grab more traces?")
            moretraces = readline()
            if moretraces == "n"
                break
            end
        end
    end
end


function filter_samples_for_task(num_dots::Int)
    human_task_directory = "/Users/nightcrawler2/humantest"
    for filename in readdir(human_task_directory)
        try
            taskfile = filename[1:length("tasktrace")+1] == string("tasktrace", num_dots)
            if !taskfile
                #loops if string is long enough (i.e. another tasktrace file) but not num_dots
                continue
            end
        catch
            #loops if string too short (i.e. a subject file)
            continue
        end
        println(filename)
        @load string(human_task_directory, "/", filename) trace_args trace_choices
        (trace, w) = Gen.generate(generate_dotmotion, trace_args, trace_choices)
        render_dotmotion(trace, true)
        println("keep trace?")
        ans1 = readline()
        if ans1 == "n"
            rm(string(human_task_directory, "/", filename))
            continue
        end
        inf_results = animate_inference(trace)
        analyze_and_plot_inference(inf_results[1:4]...)
        println("keep trace?")
        ans2 = readline()
        if ans2 == "n"
            rm(string(human_task_directory, "/", filename))
            continue
        end
    end
end    

function make_human_task(dotrange)
    human_task_directory = "/Users/nightcrawler2/humantest"
    human_task_order = []
    for filename in readdir(human_task_directory)
        for num_dots in dotrange
            try
                if filename[1:length("tasktrace")+1] == string("tasktrace", num_dots)
                    push!(human_task_order, filename)
                end
            catch
            end
        end
    end
    human_task_order = shuffle(human_task_order)
    @save string(human_task_directory, "/human_task_order.bson") human_task_order
end


function run_human_experiment()
    println("Subject ID: ")
    subject_id = readline()
    num_training_trials = 0
    partition = convert(Int, num_training_trials/3)
    training_dot_numbers = [ones(partition); 2*ones(partition);
                            3*ones(partition)]
    num_trials = 1
    human_task_directory = "/Users/nightcrawler2/humantest/"
    subject_directory = string(human_task_directory, subject_id)
    try
        mkdir(subject_directory)
    catch
    end
    biomotion_results = []
    confidence_results = []
    pw_dist_results = []
    repeats_results = []
    for training_trial in 1:num_training_trials
        num_dots = convert(Int, training_dot_numbers[training_trial])
        trace, args = dotsample(num_dots)
        pw_dist, num_repeats = render_dotmotion(trace, true)
        a_scene, confidence, biomotion = answer_portal(training_trial, subject_directory, num_dots)
    end
    @load string(human_task_directory, "human_task_order.bson") human_task_order 
    for (trial_n, trace_id) in enumerate(human_task_order)
        @load string(human_task_directory, "/", trace_id) trace_args trace_choices
        (trace, w) = Gen.generate(generate_dotmotion, trace_args, trace_choices)
        num_dots = nv(get_retval(trace)[1])
        pw_dist, num_repeats = render_dotmotion(trace, false)
#        inf_results = animate_inference(trace)
#        analyze_and_plot_inference(inf_results[1:4]...)
        # GIVES YOU SCENEGRAPH. 
# UNCOMMENT IF YOU WANT TO SHOW THE GROUND TRUTH AFTER THE FIRST PASS         
#        pw_dist, num_repeats = render_stim_only(trace, true)
        answer_graph, confidence, biomotion = answer_portal(trial_n, subject_directory, num_dots)
        savegraph(string(subject_directory, "/answers", trial_n, ".mg"), answer_graph)
        push!(biomotion_results, biomotion)
        push!(confidence_results, confidence)
        push!(pw_dist_results, pw_dist)
        push!(repeats_results, num_repeats)
    end
    @save string(subject_directory, "/biomotion.bson") biomotion_results
    @save string(subject_directory, "/confidence.bson") confidence_results
    @save string(subject_directory, "/repeats.bson") repeats_results
    @save string(subject_directory, "/pw_dist.bson") pw_dist_results
end    



function answer_portal(trial_ID::Int, directory::String, num_dots::Int)
    answer_graph = MetaDiGraph(num_dots)
    res = 1400
    stop_anim = false
    answer_scene, as_layout = layoutscene(resolution=(res, res), backgroundcolor=:black)
    dot_menus = [LMenu(answer_scene, options = ["RandomWalk", "Periodic", "Linear"]) for i in 1:num_dots]
    for (dot_id, menu) in enumerate(dot_menus)
        as_layout[dot_id, 1] = vbox!(LText(answer_scene, string("Dot ", dot_id, " Motion Type"), color=:white), menu)
    end
    tog_indices = [(dot1, dot2) for dot1 in 1:num_dots for dot2 in 1:num_dots if dot1 != dot2]
    toggles = [LToggle(answer_scene, buttoncolor=:black, active=false) for ti in tog_indices]
    toglabels = [LText(answer_scene, lift(x -> x ? string(dot1, " inherits motion of ", dot2) : string(dot1, " inherits motion of ", dot2),
                                                          toggles[i].active), color=:white) for (i, (dot1, dot2)) in enumerate(tog_indices)]
    for tog_index in 1:length(tog_indices)
        as_layout[num_dots+tog_index, 1] = hbox!(toggles[tog_index], toglabels[tog_index])
    end
    sliders = [LSlider(answer_scene, range=0:1:100, startvalue=50) for i in 1:2]
    confidence = as_layout[num_dots+length(tog_indices) + 1, 1] = vbox!(LText(answer_scene, "Confidence Level", color=:white),
                                                                        sliders[1])
    biomotion = as_layout[num_dots+length(tog_indices) + 2, 1] = vbox!(LText(answer_scene, "Biomotion Scale", color=:white),
                                                                       sliders[2])
    screen = display(answer_scene)
    stop_anim = false
    on(events(answer_scene).keyboardbuttons) do button
        if ispressed(button, Keyboard.enter)
            stop_anim = true
        end
    end
    for dot in 1:num_dots
        on(dot_menus[dot].selection) do s
            set_props!(answer_graph, dot, Dict(:MType=> s))
        end
    end
    for (tswitch, tog_inds) in enumerate(tog_indices)
        on(toggles[tswitch].active) do gt
            if gt == true
                add_edge!(answer_graph, tog_inds[2], tog_inds[1])
            else
                rem_edge!(answer_graph, tog_inds[2], tog_inds[1])
            end
        end
    end

    # on(events(answer_scene).keyboardbuttons) do button
    #     if ispressed(button, Keyboard.enter)
    #         stop_anim = true
    #     end
    # end

    # first arg passed to timedwait has to be a function
    query_enter() = stop_anim
    # will either stop when you press enter or when 30 seconds have passed. 
    timedwait(query_enter, 30.0)
#    vs = visualize_scenegraph(answer_graph)
  #  display(vs)
   # wait(Timer(callback, 5, interval=0))
    return answer_graph, sliders[1].value[], sliders[2].value[]
end    
                      



function score_human_performance(subjects::Array{String, 1}, reinfer_traces)
    human_task_directory = "/Users/nightcrawler2/humantest/"
    final_human_experiment = []
    @load string(human_task_directory, "human_task_order.bson") human_task_order
    for trace_id in human_task_order
        @load string(human_task_directory, "/", trace_id) trace_args trace_choices
        (trace, w) = Gen.generate(generate_dotmotion, trace_args, trace_choices)
        push!(final_human_experiment, trace)
    end
    if reinfer_traces
        inf_results_importance_w_hyper = [animate_inference(trace) for trace in final_human_experiment]
        inf_results_enumeration = [bayesian_observer(trace)[2] for trace in final_human_experiment]
        inf_results_enumeration_args_and_choices = [[get_args(ir), get_choices(ir)] for ir in inf_results_enumeration]
        @save string(human_task_directory, "inference_results_imp.bson") inf_results_importance_w_hyper
        @save string(human_task_directory, "inference_results_enum.bson") inf_results_enumeration_args_and_choices
    else
        @load string(human_task_directory, "inference_results_imp.bson") inf_results_importance_w_hyper
        @load string(human_task_directory, "inference_results_enum.bson") inf_results_enumeration_args_and_choices
        inf_results_enumeration = [Gen.generate(generate_dotmotion, tr[1], tr[2])[1] for tr in inf_results_enumeration_args_and_choices]
    end
    inf_results_importance = [inf_res[1:4] for inf_res in inf_results_importance_w_hyper]        
    hyperparams = [hyperparameter_inference(inf_res[end], trace)
                   for (inf_res, trace) in zip(inf_results_importance_w_hyper, final_human_experiment)]
    # have to make sure the hyperparam and gt answers are in equivalent form so they can be compared.
    # dont need the full results here. just need the top graph. 
    all_subject_results = []
    for subject in subjects
        directory = string(human_task_directory, subject)
        @load string(directory, "/biomotion.bson") biomotion_results
        @load string(directory, "/confidence.bson") confidence_results
        @load string(directory, "/repeats.bson") repeats_results
        @load string(directory, "/pw_dist.bson") pw_dist_results
        trial_results = Dict(:subject => subject,
                             :human_truth_match => [],
                             :truth_importance_match => [],
                             :truth_enum_match => [],
                             :human_importance_match => [],
                             :human_enum_match => [])
        for (tr, trace)  in enumerate(final_human_experiment)
            answer_graph = loadgraph(string(directory, "/answers", tr, ".mg"), MGFormat())
            # run inference here on the saved traces
            top_importance_hit = analyze_inference_results(inf_results_importance[tr]...)[1][1]
            top_enumeration_hit = get_retval(inf_results_enumeration[tr])[1]
            push!(trial_results[:human_truth_match], compare_scenegraphs(answer_graph, get_retval(trace)[1]))
            push!(trial_results[:human_importance_match], compare_scenegraphs(answer_graph, top_importance_hit))
            push!(trial_results[:human_enum_match], compare_scenegraphs(answer_graph, top_enumeration_hit))
            push!(trial_results[:truth_enum_match], compare_scenegraphs(get_retval(trace)[1], top_enumeration_hit))
            push!(trial_results[:truth_importance_match], compare_scenegraphs(get_retval(trace)[1], top_importance_hit))
        end
        push!(all_subject_results, trial_results)
    end
    return all_subject_results
end

function plot_subject_performance(subject_results_dict)
    scene = Scene()
    collect_human_scores = [map(x-> float(all(x)), dict[:human_truth_match]) for dict in subject_results_dict]
    collect_imp_scores = [map(x-> float(all(x)), dict[:truth_importance_match]) for dict in subject_results_dict]
    collect_enum_scores = [map(x-> float(all(x)), dict[:truth_enum_match]) for dict in subject_results_dict]
    human_scores_by_trial = [collect(z) for z in zip(collect_human_scores...)]
    imp_scores_by_trial = [collect(z) for z in zip(collect_imp_scores...)]
    enum_scores_by_trial = [collect(z) for z in zip(collect_enum_scores...)]
#    println(collect_human_scores)
 #   println(collect_imp_scores)
    println(collect_enum_scores)
    # boxplot works where the first array is a set of groups and the second array are the values assigned to the groups:
    # i.e.  [1,2, 1], [4,5,5] will have 4 and 5 in group 1 and 5 in group2
    trial_indices = [ind*ones(length(sc)) for (ind, sc) in enumerate(human_scores_by_trial)]
    println(trial_indices)
    boxplot_entries = [vcat(trial_indices...), vcat(human_scores_by_trial...)]
    # THIS IS IF YOU WANT A BOXPLOT FOR EACH TRIAL FOR EACH CONDITION. 
    boxplot!(scene, boxplot_entries..., width=.1)
    boxplot!(scene, .1 .+ boxplot_entries[1], boxplot_entries[2], width=.1, color=:lightblue)
    ylims!(scene, 0, 2)
    display(scene)
    return human_scores_by_trial, imp_scores_by_trial, enum_scores_by_trial
end    

        
function compare_scenegraphs(mg1::MetaDiGraph{Int64, Float64},
                             mg2::MetaDiGraph{Int64, Float64})
    # for a systemic analysis, want
    # 1) correct vs incorrect graph
    # 2) correct vs incorrect motion types
    # 3) graph inversions (i.e. correct grouping, wrong direction)
    # 4) motion inversions (
    # 5) half correct / one third correct motion patterns
    
    motion_types_correct = [props(mg1, dot)[:MType] == props(mg2, dot)[:MType] for dot in 1:nv(mg1)]
    scenegraph_correct = mg1.graph == mg2.graph
    scenegraph_inverted = mg1.graph == reverse(mg2.graph)
    return all(motion_types_correct), scenegraph_correct
end    


# suppose you want to make a proposal to twiddle perceptual noise on a dot. you can't just do it
# b/c you have to make the corresponding change to x_observable. still possible if you write
# reversible jump. you may have an extremely low prob of acceptance b/c of the narrow window of x_obs. oh you have to assure that x_obs and y_obs remain the same and have to construct it using changes or twiddles in other variables that build them. make some combination of changes that results in the same value of x_obs.

# calling Gen.update when you're not changing the x_observable is going to
# comparisons would no longer be meaningful -- huge penalties from the way x_observable is constructed. 

# note the final mvnormal draw does nothing. the x_timeseries and x_observable are identical.

# OK YOUR NEW INFRASTRUCTURE WILL USE UPDATE AND GENERATE. YOUR GENERATE CALLS WILL REMOVE THE OBSERVABLE
# DRAWS FROM THE CHOICEMAP, AND ALSO THE DRAWS OF WHATEVER YOU ARE REPLACING. YOUR UPDATE CALLS WILL
# SIMPLY REPLACE THE VALUES. THEN YOU WILL ALWAYS PLOT X_OBSERVABLE. GET RID OF TRACE TO TREE, AND ANY TREE REQUIRING CALLS
# GENERATE WILL ALSO GIVE BACK UPDATED TREES. UPDATE WILL NOT (ONLY FOR THE CHANGED VARIABLE). 
