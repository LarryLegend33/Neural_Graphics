using Makie
using AbstractPlotting
using StatsMakie
using MakieLayout
using Gen
using LinearAlgebra
using LightGraphs
using MetaGraphs
using Random
using Images
using ShiftedArrays
using ColorSchemes
using Statistics
using StatsBase
using CurricularAnalytics
using BSON: @save, @load
using MappedArrays


#- starting with init positions b/c this is the type of custom proposal you will get from the tectum. you won't get offests for free. this model accounts for distance effects and velocity effects by traversing the tree. 

#want a balance between inferability and smoothness
framerate = 60
time_duration = 10
num_velocity_points = time_duration * 4

# filling in n-1 samples for every interpolation, where n is the
# length of the velocity vector. your final amount of samples doubles this each time, then adds 1. 
interp_iters = round(Int64, log(2, (framerate * time_duration) / (num_velocity_points -1)))

function interpolate_coords(vel, iter)
    if iter == 0
        return vel
    else
        interped_vel = vcat([[vel[i], mean([vel[i], vel[i+1]])] for i in 1:length(vel)-1]...)
        push!(interped_vel, vel[end])
        interpolate_coords(vcat(interped_vel...), iter-1)        
    end
end


""" GENERATIVE CODE: These functions output dotmotion stimuli using GP-generated timeseries"""

@gen function populate_edges(motion_tree::MetaDiGraph{Int64, Float64},
                             candidate_pairs::Array{Tuple, 1})
    if isempty(candidate_pairs)
        return motion_tree
    end
    (current_dot, cand_parent) = first(candidate_pairs)
    current_paths = all_paths(motion_tree)
    # prevents recursion in the motion tree. can add this back in someday to model non-rigid bodies
    # like a ring of beads
    if any([current_dot in path && cand_parent in path for path in current_paths])        
        add_edge = { (:edge, cand_parent, current_dot) } ~  bernoulli(0)
    else
        if isempty(inneighbors(motion_tree, cand_parent))
            add_edge = { (:edge, cand_parent, current_dot) } ~  bernoulli(.3)
        else
            add_edge = { (:edge, cand_parent, current_dot) } ~  bernoulli(.1)
        end
    end
    if add_edge
        add_edge!(motion_tree, cand_parent, current_dot)
    end
    {*} ~ populate_edges(motion_tree, candidate_pairs[2:end])
end

# note that the graphs can all be mutated. if your arg set is constant, it will still be manipulated if it was created
# as a variable. declared arg variables mutate inside a generative function.

# note that if you constrain generate_dotmotion on an unallowable edge (e.g. [1,3]), it wont prevent the inverse edge from being true.
# have to specify all edges at once. 

# make sure to arrange dots before entering assign_positions_and_velocities function
# have to specify position and velocity of all parents first b/c child nodes depend on it.
# arranging by number of inneighbors first and outneighbors second (inverted) guarantees parents
# are specified before children. 

@gen function generate_dotmotion(ts::Array{Float64}, 
                                 n_dots::Int)
    motion_tree = MetaDiGraph(n_dots)
    order_distribution = return_dot_distribution(n_dots)
    perceptual_order = { :order_choice } ~ order_distribution()
    # noise = { :noise } ~ gamma_bounded_below(.1, .1, .0005)
    noise = { :noise } ~ multinomial(param_dict[:noise])
    candidate_edges = [p for p in Iterators.product(perceptual_order, perceptual_order) if p[1] != p[2]]
    motion_tree_updated = {*} ~ populate_edges(motion_tree, candidate_edges)
    dot_list = sort(collect(1:nv(motion_tree_updated)),
                    by=ϕ->(size(inneighbors(motion_tree_updated, ϕ))[1]))
    motion_tree_assigned = {*} ~ assign_positions_and_velocities(motion_tree_updated,
                                                                 dot_list,
                                                                 ts,
                                                                 noise)
    return motion_tree_assigned, dot_list
end


@gen function assign_positions_and_velocities(motion_tree::MetaDiGraph{Int64, Float64},
                                              dots::Array{Int64}, ts::Array{Float64}, noise::Float64)
    if isempty(dots)
        return motion_tree
    else
        dot = first(dots)
        parents = inneighbors(motion_tree, dot)
        # if parent values haven't been assigned yet, put the dot at the end and recurse.
        if count(iszero, [props(motion_tree, p).count for p in parents]) != 0
            {*} ~ assign_positions_and_velocities(motion_tree, [dots[2:end];dot], ts, noise)
        else
            # use for flat prior on position
            start_x = {(:start_x, dot)} ~ uniform(-5, 5)
            start_y = {(:start_y, dot)} ~ uniform(-5, 5)
            x_vel_mean = zeros(length(ts))
            y_vel_mean = zeros(length(ts))

            if isempty(parents)
                #uncomment to use biased prior on initial position
    #            start_x = {(:start_x, dot)} ~ uniform_discrete(-5, 5)
    #            start_y = {(:start_y, dot)} ~ uniform_discrete(-5, 5)
                x_vel_mean = zeros(length(ts))
                y_vel_mean = zeros(length(ts))
            else
                if size(parents)[1] > 1
                    avg_parent_position = mean([props(motion_tree, p)[:Position] for p in parents])
                    parent_position = [round(Int, pp) for pp in avg_parent_position]
                else
                    parent_position = props(motion_tree, parents[1])[:Position]
                end
     #           start_x = {(:start_x, dot)} ~ uniform_discrete(parent_position[1]-1, parent_position[1]+1)
    #            start_y = {(:start_y, dot)} ~ uniform_discrete(parent_position[2]-1, parent_position[2]+1)
                parent_velocities_x = [props(motion_tree, p)[:Velocity_X] for p in parents]
                parent_velocities_y = [props(motion_tree, p)[:Velocity_Y] for p in parents]
            end

            if !isempty(parents)
                if size(parents)[1] == 1
                    x_vel_mean = parent_velocities_x[1]
                    y_vel_mean = parent_velocities_y[1]
                else
                    x_vel_mean = sum(parent_velocities_x)
                    y_vel_mean = sum(parent_velocities_y)
                end
            end
            # sample a kernel type for the dot here. then assign with cov prior conditioned on type
            kernel_type = {(:kernel_type, dot)} ~ choose_kernel_type()
            cov_func_x = {*} ~ covariance_prior(kernel_type, dot, 1)
            cov_func_y = {*} ~ covariance_prior(kernel_type, dot, 2)
            covmat_x = compute_cov_matrix_vectorized(cov_func_x, noise, ts)
            covmat_y = compute_cov_matrix_vectorized(cov_func_y, noise, ts)
            x_vel = {(:x_vel, dot)} ~ mvnormal(x_vel_mean, covmat_x)
            y_vel = {(:y_vel, dot)} ~ mvnormal(y_vel_mean, covmat_y)
            # Sample from the GP using a multivariate normal distribution with
            # the kernel-derived covariance matrix.
            set_props!(motion_tree, dot,
                       Dict(:Position=>[start_x, start_y], :Velocity_X=>x_vel, :Velocity_Y=>y_vel, :MType=>string(typeof(cov_func_x))))
            {*} ~ assign_positions_and_velocities(motion_tree, dots[2:end], ts, noise)
        end
    end
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


""" These functions wrap model runs and inference to evaluate inference accuracy """ 
function dotsample(num_dots::Int)
    ts = range(1, stop=time_duration, length=num_velocity_points)
    gdm_args = (convert(Array{Float64}, ts), num_dots)
    trace = Gen.simulate(generate_dotmotion, gdm_args)
    trace_choices = get_choices(trace)
    return trace, gdm_args
end    

function dotwrap(num_dots::Int)
    trace, args = dotsample(num_dots)
    pw_distances, number_repeats, visible_parent = render_stim_only(trace, true)
#    inf_results, top_bayes_graph = bayesian_observer(trace)
#    inf_results = animate_inference(trace)
#    analyze_and_plot_inference(inf_results[1:4]...)
 #   analyze_and_plot_inference(inf_results...)
    #   return trace, inf_results, pw_distances, number_repeats, visible_parent
#    return trace, inf_results, top_bayes_graph
 #   return trace, inf_results
end


    

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
        render_stim_only(trace, true)
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
            @save string(human_task_directory,"/", trace_file_id) trace
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
        @load string(human_task_directory, "/", filename) trace
        render_stim_only(trace, true)
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
    visible_parent_results = []
    for training_trial in 1:num_training_trials
        num_dots = convert(Int, training_dot_numbers[training_trial])
        trace, args = dotsample(num_dots)
        pw_dist, num_repeats, visible_parent = render_stim_only(trace, true)
        a_scene, confidence, biomotion = answer_portal(training_trial, subject_directory, num_dots)
    end
    @load string(human_task_directory, "human_task_order.bson") human_task_order 
    for (trial_n, trace_id) in enumerate(human_task_order)
        @load string(human_task_directory, "/", trace_id) trace
        num_dots = nv(get_retval(trace)[1])
        pw_dist, num_repeats, visible_parent = render_stim_only(trace, false)
        answer_graph, confidence, biomotion = answer_portal(trial_n, subject_directory, num_dots)
        savegraph(string(subject_directory, "/answers", trial_n, ".mg"), answer_graph)
        push!(biomotion_results, biomotion)
        push!(confidence_results, confidence)
        push!(pw_dist_results, pw_dist)
        push!(repeats_results, num_repeats)
        push!(visible_parent_results, visible_parent)
    end
    @save string(subject_directory, "/biomotion.bson") biomotion_results
    @save string(subject_directory, "/confidence.bson") confidence_results
    @save string(subject_directory, "/repeats.bson") repeats_results
    @save string(subject_directory, "/pw_dist.bson") pw_dist_results
    @save string(subject_directory, "/visible_parent.bson") visible_parent_results
end    



function answer_portal(trial_ID::Int, directory::String, num_dots::Int)
    answer_graph = MetaDiGraph(num_dots)
    res = 1400
    answer_scene, as_layout = layoutscene(resolution=(res, res), backgroundcolor=:black)
    dot_menus = [LMenu(answer_scene, options = ["RandomWalk", "Periodic", "UniformLinear"]) for i in 1:num_dots]
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
    callback(timer) = (println("times up"))
    wait(Timer(callback, 20, interval=0))
    vs = visualize_scenegraph(answer_graph)
  #  display(vs)
   # wait(Timer(callback, 5, interval=0))
    return answer_graph, sliders[1].value[], sliders[2].value[]
end    
                      


#Friday. Make a human experiment with 4 or 5 trials. Make sure it launches correctly.
#score performance here and make a graph that follows the RDD. 

function score_performance(subjects::Array{String, 1})

    humantask_directory = "/Users/nightcrawler2/humantest/"
    final_human_experiment = []
    @load string(humantask_directory, "human_task_order.bson") human_task_order
    for trace_id in human_task_order
        @load string(humantask_directory, trace_id) trace
        push!(final_human_experiment, trace)
    end
    inf_results_importance_w_hyper = [animate_inference(trace) for trace in final_human_experiment]
    inf_results_importance = [inf_res[1:4] for inf_res in inf_results_importance_w_hyper]
    hyperparams = [hyperparameter_inference(inf_res[end], trace)
                   for (inf_res, trace) in zip(inf_results_importance_w_hyper, final_human_experiment)]
    # have to make sure the hyperparam and gt answers are in equivalent form so they can be compared.

    # dont need the full results here. just need the top graph. 
    inf_results_enumeration = [bayesian_observer(trace) for trace in final_human_experiment]
    all_subject_results = []
    for subject in subjects
        directory = string("/Users/nightcrawler2/humantest/", subject)
        @load string(directory, "/biomotion.bson") biomotion_results
        @load string(directory, "/confidence.bson") confidence_results
        @load string(directory, "/repeats.bson") repeats_results
        @load string(directory, "/pw_dist.bson") pw_dist_results
        @load string(directory, "/visible_parent.bson") visible_parent_results
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
            top_enumeration_hit = get_retval(inf_results_enumeration[tr][2])[1]
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

    println(collect_human_scores)
    println(collect_imp_scores)
    # boxplot works where the first array is a set of groups and the second array are the values assigned to the groups:
    # i.e.  [1,2, 1], [4,5,5] will have 4 and 5 in group 1 and 5 in group2
    trial_indices = [ind*ones(length(sc)) for (ind, sc) in enumerate(human_scores_by_trial)]
    boxplot_entries = [vcat(trial_indices...), vcat(human_scores_by_trial...)]

    # THIS IS IF YOU WANT A BOXPLOT FOR EACH TRIAL FOR EACH CONDITION. 
    #    boxplot!(scene, boxplot_entries..., width=.1)
    #    boxplot!(scene, .1 .+ boxplot_entries[1], boxplot_entries[2], width=.1, color=:lightblue)

    
    

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
    kernel_combos = collect(Iterators.product(kernel_choices...))
    possible_edges = [(i, j) for i in 1:num_dots for j in 1:num_dots if i != j]
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
    for eg in edge_truthtable
        enum_constraints = Gen.choicemap()
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
            for i in 1:num_dots
                enum_constraints[(:x_vel, i)] = trace[(:x_vel, i)]
                enum_constraints[(:y_vel, i)] = trace[(:y_vel, i)]
                enum_constraints[(:start_x, i)] = trace[(:start_x, i)]
                enum_constraints[(:start_y, i)] = trace[(:start_y, i)]
            end
            # clears order from the original trace
            constraints_no_noise_order = [map_entry for map_entry in get_values_shallow(enum_constraints) if !(typeof(map_entry[1]) == Symbol)]
            constraints_no_params = [map_entry for map_entry in constraints_no_noise_order if !(map_entry[1][1] in [:amplitude, :variance, :covariance, :period, :lengthscale])]
            enum_constraints = Gen.choicemap(constraints_no_params...)
            kernel_assignments = [enum_constraints[(:kernel_type, i)] for i in 1:num_dots]
            collect_params_per_ktype = [Iterators.product([param_dict[ktype]..., param_dict[ktype]...]...) for ktype in kernel_assignments]
            all_param_permutations = Iterators.product(collect_params_per_ktype...)
            # this is totally correct. for random random, gives all combinations in form of ((10, 10), (10, 11))
            sum_order_scores = 0
            for dp in all_dot_permutations(num_dots)
                enum_constraints[:order_choice] = dp
                sum_order_scores += .001
                #    (new_trace, w, a, ad) = Gen.update(trace, get_args(trace), (NoChange(),), enum_constraints)
                #     pscore = exp(get_score(new_trace))
 #               sum_order_scores += exp(w)
                for noise in param_dict[:noise]
                    enum_constraints[:noise] = noise
                    for param_permutation in all_param_permutations
                        param_assignments = hyper_permutation_to_assignment(kernel_assignments, param_permutation)
                        for p in param_assignments
                            enum_constraints[p.first] = p.second
                        end
                        (tr, w) = Gen.generate(generate_dotmotion, trace_args, enum_constraints)
                        if w > top_score
                            top_trace = tr
                            top_score = w
                            println("top score")
                            println(top_score)
                        end
                     end
                end
            end
            append!(scores, sum_order_scores)
        end
    end
  #  scores /= sum(scores)
    score_matrix = reshape(scores, prod(collect(size(kernel_combos))), size(edge_truthtable)[1])
    plotvals = [score_matrix, kernel_combos, possible_edges, edge_truthtable]
    return plotvals, top_trace
end


function animate_inference(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    num_dots = nv(get_retval(trace)[1])
    kernel_choices = [kernel_types for i in 1:num_dots]
    kernel_combos = collect(Iterators.product(kernel_choices...))
    possible_edges = [(i, j) for i in 1:num_dots for j in 1:num_dots if i != j]
    truth_entry = [[0,1] for i in 1:size(possible_edges)[1]]
    all_samples, edge_samples, vel_samples = imp_inference(trace)
    joint_edge_vel = [(Tuple(e), Tuple(v)) for (e,v) in zip(edge_samples, vel_samples)]
    
    # filters trees with n_dot or more edges
    if !isempty(truth_entry)
        unfiltered_truthtable = [j for j in Iterators.product(truth_entry...) if sum(j) < num_dots]
        edge_truthtable = loopfilter(possible_edges, unfiltered_truthtable)
    else
        unfiltered_truthtable = edge_truthtable = [()]
    end
    importance_counts = []
    # creates a list with entries that look like this ((1, 0), (UniformLinear, RandomWalk)), where
    # each entry is an importance sample
    for eg in edge_truthtable
        for kc in kernel_combos
            ev_count = count(λ -> (λ[1] == eg && λ[2] == kc), joint_edge_vel)
            push!(importance_counts, ev_count)
        end
    end
    count_matrix = reshape(importance_counts, prod(collect(size(kernel_combos))), size(edge_truthtable)[1])
    inf_results = [count_matrix, kernel_combos, possible_edges, edge_truthtable, all_samples]
#    plot_inference_results(inf_results...)
    return inf_results
end

function imp_inference(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    trace_choices = get_choices(trace)
    args = get_args(trace)
    observation = Gen.choicemap()
    all_samples = []
    num_dots = nv(get_retval(trace)[1])
    num_particles = (num_dots ^ 2) * 100
    num_resamples = 30
    for i in 1:num_dots
        observation[(:x_vel, i)] = trace[(:x_vel, i)]
        observation[(:start_y, i)] = trace[(:start_y, i)]
        observation[(:y_vel, i)] = trace[(:y_vel, i)]
        observation[(:start_x, i)] = trace[(:start_x, i)]        
    end
    edge_list = []
    kernel_types = []
    for i in 1:num_resamples
        (tr, w) = Gen.importance_resampling(generate_dotmotion, args, observation, num_particles)
        push!(edge_list, [tr[(:edge, j, k)] for j in 1:num_dots for k in 1:num_dots if j!=k])
        push!(kernel_types, [tr[(:kernel_type, j)] for j in 1:num_dots])
        push!(all_samples, tr)
#        s = visualize_scenegraph(get_retval(tr)[1])
 #       display(s)
    end
    return all_samples, edge_list, kernel_types
end


function analyze_and_plot_inference(score_matrix::Array{Any, 2}, kernels,
                                    possible_edges, edge_truth)
    graphs_and_probs = analyze_inference_results(score_matrix, kernels, possible_edges, edge_truth)
    plot_inference_results(graphs_and_probs[2:end]...)
end    


function analyze_inference_results(score_matrix::Array{Any, 2}, kernels, possible_edges, edge_truth)
    # TOP 3 GRAPHS
    top_graphs = find_top_n_props(3, score_matrix, [])
    rendered_graphs = []
    probabilities = []
    top_metagraph_scenes = []
    edge_combinations = [[e_entry for (i, e_entry) in enumerate(possible_edges) if et[i] == 1] for et in edge_truth]
    # if you print out the score matrix, doesn't equal number of samples requested
    for tg in top_graphs
        score_index = tg[2].I
        score = tg[1]
        push!(probabilities, score)
        vel_types = kernels[score_index[1]]
        edges = edge_combinations[score_index[2]]
        top_g = MetaDiGraph(length(vel_types))
        for edge in edges
            add_edge!(top_g, edge[1], edge[2])
        end
        for (node, vel) in enumerate(vel_types)
            set_props!(top_g, node, Dict(:MType=>string(vel)))
        end
        viz_graph = visualize_scenegraph(top_g)
        push!(top_metagraph_scenes, top_g)
        push!(rendered_graphs, viz_graph)
    end
    sum_probs = sum(mappedarray(x-> isfinite(x) ? x : 0, score_matrix))
    return top_metagraph_scenes, rendered_graphs, probabilities ./ sum_probs
end

function plot_inference_results(rendered_graphs, probabilities)
    scene, layout = layoutscene(resolution=(300, 300), backgroundcolor=RGBf0(0, 0, 0))
    white = RGBf0(255,255,255)
    black = RGBf0(0,0,0)
    gray = RGBf0(100, 100, 100)
    axes = LAxis(scene, backgroundcolor=black, ylabelcolor=white, xticklabelcolor=black, yticklabelcolor=white, 
                 xtickcolor=white, ytickcolor=white, xgridcolor=black, ygridcolor=gray,
                 xticklabelrotation = pi/2,  xticklabelalign = (:top, :top), yticklabelalign = (:top, :top))
    layout[1, 1] = axes
    barwidth = (.2 / 3) * length(probabilities)
    scene_graph_scene = vbox(rendered_graphs...)
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
    bp = barplot!(axes, bar_x, convert(Array{Float64, 1}, probabilities),
                  color=:white,
                  backgroundcolor=:black, width=barwidth)

    axes.ylabel = "Posterior Probability"
    limits!(axes, xaxlims)
    final_scene = hbox(scene, 
                       scene_graph_scene)
    screen = display(final_scene)
    # uncomment if you want to block until the window is closed
    #  wait(screen)
    
    # OLD LOUSLY LOOKING HEATMAP
    # yticklabs = [string(ec) for ec in edge_combinations]
    # xticks = (0:prod(collect(size(kernels)))-1, [string([string(ks)[1] for ks in k]...) for k in kernels])
    # yticks = (1:size(edge_truth)[1], [yt[1] != 'T' ? yt : "[]" for yt in yticklabs])
    # hm = heatmap!(axes, score_matrix, colormap=:viridis)
    # layout[1,1] = axes
    # axes.xticks = xticks
    # axes.yticks = yticks
    # hm_sublayout = GridLayout()
    # layout[1, 1] = hm_sublayout
    # cbar = hm_sublayout[:, 2] = LColorbar(scene, hm, width=14, height=Relative(.91), label = "Probability", labelcolor=white, tickcolor=black, labelsize=10)

end


# note for JM slides, used 20 particles for 2 dots, 100 for 3.
# make imp_inference take a number of particles. 



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
        elseif kernels[dot] == UniformLinear
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
        elseif choices[(:kernel_type, n)] == UniformLinear
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




"""Create Makie Rendering Environment"""

function tree_to_coords(tree::MetaDiGraph{Int64, Float64})
    num_dots = nv(tree)
    dotmotion = fill(zeros(2), num_dots, size(interpolate_coords(props(tree, 1)[:Velocity_X], interp_iters))[1])
    # Assign first dot positions based on its initial XY position and velocities
    for dot in 1:num_dots
        dot_data = props(tree, dot)
        dotmotion[dot, :] = [[x, y] for (x, y) in zip(
            dot_data[:Position][1] .+ cumsum(interpolate_coords(dot_data[:Velocity_X], interp_iters)) ./ framerate,
            dot_data[:Position][2] .+ cumsum(interpolate_coords(dot_data[:Velocity_Y], interp_iters)) ./ framerate)]
    end
    dotmotion_tuples = [[Tuple(dotmotion[i, j]) for i in 1:num_dots] for j in 1:size(dotmotion)[2]]
    return dotmotion_tuples, dotmotion
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

        # inverted this at the end! wtf just make it right here. 
        
        [yc[dot] = longest_path - length(reachable_to(motion_tree.graph, dot)) for dot in path]
        xy_node_positions(paths[2:end], xc, yc, n_iters+1, motion_tree, longest_path)
    end
end


function visualize_scenegraph(motion_tree::MetaDiGraph{Int64, Float64})
    outer_padding = 0
    res = 1000
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
    
    # create scene without layout b/c text only works in scenes -- can't add it to LAxis.
    scene = Scene(backgroundcolor=RGBf0(0, 0, 0), resolution=(800,800))
    for e in edges(motion_tree)
        arrows!(scene, [node_xs[e.src]], [node_ys[e.src]],
                .8 .* [node_xs[e.dst]-node_xs[e.src]], .8 .* [node_ys[e.dst]-node_ys[e.src]], arrowcolor=:lightgray, linecolor=:lightgray, arrowsize=.1)
    end
    
    for v in 1:nv(motion_tree)
        mtype = props(motion_tree, v)[:MType]
        if mtype == "UniformLinear"
            nodecolor = :lightgreen
        elseif mtype == "RandomWalk"
            nodecolor = :orange
        elseif mtype == "Periodic"
            nodecolor = :skyblue
        elseif mtype == "AccelLinear"
            nodecolor = :pink
        end
        scatter!(scene, [(node_xs[v], node_ys[v])], markersize=50px, color=nodecolor)
        text!(scene, string(v), position=(node_xs[v], node_ys[v]), align= (:center, :center),
              textsize=.2, color=:black, overdraw=true)
    end
    #    limits!(scene, BBox(0, xbounds, 0, ybounds))
    xlims!(scene, 0, xbounds)
    ylims!(scene, 0, ybounds)
#    display(scene)
    return scene
end

# may be a good idea to use vbox and hbox instead of layout. I like it.
# if you want to show 3 side by side graphs, use vbox of each scene.



    # for each vertex, count number of incoming edges.
    # use "reachable_to" or "reachable_from" if graph is 1->2->3, reachable_from(g, 1) = [2, 3]
    
    # longest_pathlen describes the height. should be this length plus 2 (one slot at top and bottom free).
    # number of paths total should be x 





function find_top_n_props(n::Int,
                          score_matrix::Array{Any, 2},
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

# problem is if you eliminate it and its still the max (i.e. once you eliminate it, everything else is 0

function render_stim_only(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}}, show_scenegraph::Bool)
    motion_tree = get_retval(trace)[1]
    # ask which has the max of incoming edges
    bounds = 25
    res = 1400
    outer_padding = 0
    dotmotion, raw_dotmotion = tree_to_coords(motion_tree)
    dot_w_most_incoming_edges = findmax(map(x -> length(reachable_to(motion_tree.graph, x)), 1:nv(motion_tree)))[2]    
    if bernoulli(.05) && ne(motion_tree) > 0 
        parent_visible = false
        dotmotion = [[d[dot_w_most_incoming_edges]] for d in dotmotion]
        println("invisible parent")
    else
        parent_visible = true
    end
    pairwise_distances = calculate_pairwise_distance(raw_dotmotion)
    stationary_duration = 100
    stationary_coords = [dotmotion[1] for i in 1:stationary_duration]
    f(t, coords) = coords[t]
    f_color(t) = t < stationary_duration ? :white : :black
    n_rows = 3
    n_cols = 2
    white = RGBf0(255,255,255)
    black = RGBf0(0,0,0)
    scene = Scene(backgroundcolor=black, resolution=(res, res))
    time_node = Node(1);
    f(t, coords) = coords[t]
    for n in 1:nv(motion_tree)
        textloc = Tuple(props(motion_tree, n)[:Position])
        text!(scene, string(n), position = textloc .+ .1, color=lift(t -> f_color(t), time_node), textsize=2.5)
    end
    scatter!(scene, lift(t -> f(t, [stationary_coords; dotmotion]), time_node), markersize=15px, color=RGBf0(255, 255, 255))
    xlims!(scene, (-bounds, bounds))
    ylims!(scene, (-bounds, bounds))
    # for j in 1:nv(motion_tree)
    #     println(trace[(:kernel_type, j)])
    # end
    # Uncomment if you want to visualize scenegraph side by side with stimulus
    if show_scenegraph
        gscene = visualize_scenegraph(motion_tree)
        gt_scene = vbox(scene, gscene)
        screen = display(gt_scene)
    else
        screen = display(scene)
    end
    #    record(gt_scene, "stimulus.mp4", 1:size(dotmotion)[1]; framerate=60) do i
    #    for i in 1:size(dotmotion)[1]
    i = 0
    num_repeats = 0
    #    isopen(scene))
    stop_anim = false
    on(events(scene).keyboardbuttons) do button
        if ispressed(button, Keyboard.enter)
            stop_anim = true
        end
    end
    while(!stop_anim)
        i += 1
        if i == size([stationary_coords; dotmotion])[1]
            i = 1
            num_repeats += 1
        end
        time_node[] = i
        sleep(1/framerate)
    end
    return pairwise_distances, num_repeats, parent_visible
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

    
"""Uniform Linear Kernel"""
struct UniformLinear <: PrimitiveKernel
    param::Float64
end

eval_cov(node::UniformLinear, t1, t2) = node.param


function eval_cov_mat(node::UniformLinear, ts::Array{Float64})
    n = length(ts)
    fill(node.param, (n, n))
end





"""Linear kernel"""
struct AccelLinear <: PrimitiveKernel
    param::Float64
end

eval_cov(node::AccelLinear, t1, t2) = (t1 - node.param) * (t2 - node.param)

function eval_cov_mat(node::AccelLinear, ts::Array{Float64})
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

# This is an array of data types. Each data type takes a parameter, and each data type has a multiple dispatch
# call associated with it to create a covariance matrix. 
    
kernel_types = [RandomWalk, UniformLinear, Periodic]
@dist choose_kernel_type() = kernel_types[categorical([1/3, 1/3, 1/3])]

#kernel_types = [RandomWalk, UniformLinear, Periodic, Stationary]
#@dist choose_kernel_type() = kernel_types[categorical([1/4, 1/4, 1/4, 1/4])]


function all_dot_permutations(n_dots)
    all_ranges = [1:n_dots for i in 1:n_dots]
    all_permutations = [i for i in Iterators.product(all_ranges...) if length(unique(i)) == n_dots]
    return collect(all_permutations)
end    

function return_dot_distribution(n_dots)
    d_permut = all_dot_permutations(n_dots)
    @dist dot_permutations() = d_permut[categorical([1/length(d_permut) for i in 1:length(d_permut)])]
end    






# I tested this function under Gen.generate and constrained choices of kernel types
# unconstrained, weight is correctly 0. constrained, weights are identical to categorial probabilities
# returns a natural log of the prob. 

@gen function covariance_simple(kernel_type, dot, dim)
    if kernel_type == Periodic
        #        kernel_args = [.5, .5]
        # note the velocity profile is updating at 4Hz (40 samples over 10 sec),
        # so have to have period be factor of 4 to look periodic. 
        kernel_args = [6, .5, .3]

#        kernel_args = [20, 20]
    elseif kernel_type == UniformLinear
        # use 1 if bounds are 10, 2 if 20
        kernel_args = [4]
#        kernel_args = [3]
    elseif kernel_type == AccelLinear
        kernel_args = [.15]
    elseif kernel_type == RandomWalk
#        kernel_args = [.2]
        kernel_args = [20]
    else
        kernel_args = [1]
    end
    return kernel_type(kernel_args...)
end 


# @gen function covariance_prior(kernel_type, dot, dim)
#     if kernel_type == Periodic
#         kernel_args = [{(:scale, dot, dim)} ~ uniform(2, 5),
#                        {(:length, dot, dim)} ~ uniform(.5, 2),
#                        {(:period, dot, dim)} ~ uniform(.1, 3)]
#     elseif kernel_type == UniformLinear
#         kernel_args = [{(:param, dot, dim)} ~ uniform(0, 30)]
#     elseif kernel_type == RandomWalk
#         kernel_args = [{(:param, dot, dim)} ~ uniform(0, 30)]
#     else
#         kernel_args = [{(:param, dot, dim)} ~ uniform(0, 1)]
#     end
#     return kernel_type(kernel_args...)
# end





# importance sampling works here, but perceptually random and periodic differ too much
# if the low bound of sinusoidal is 1. its much smoother than random. dropping the subsampling
# unsmoothens periodic, but unsmoothens random to the point where you can clearly tell just by smoothness. 

# @gen function covariance_prior(kernel_type, dot, dim)
#     # Choose a type of kernel
#     # If this is a composite node, recursively generate subtrees. For now, too complex. 
#     # if in(kernel_type, [Plus, Times])
#     #     return kernel_type({ :left } ~ covariance_prior(), { :right } ~ covariance_prior())
#     # end
#     # Otherwise, generate parameters for the primitive kernel.
#     if kernel_type == Periodic
#         kernel_args = [{(:amplitude, dot, dim)} ~ uniform_discrete(2, 6),
#                        {(:lengthscale, dot, dim)} ~ multinomial([.5, 1, 2]),
#                        {(:period, dot, dim)} ~ multinomial([n for n in -1:.25:1 if n != 0])]
#     elseif kernel_type == UniformLinear
#         kernel_args = [{(:covariance, dot, dim)} ~ uniform_discrete(0, 30)]
#     elseif kernel_type == RandomWalk
#         kernel_args = [{(:variance, dot, dim)} ~ uniform_discrete(10, 30)]
#     end
#     return kernel_type(kernel_args...)
# end


# param_dict = Dict(Periodic => [[2, 4, 6], [.5, 1, 2], .2:.2:1],
#                   RandomWalk => [collect(1:45)],
#                   UniformLinear => [collect(1:45)])


# always make sure length scale is small. more dynamics.
# 
param_dict = Dict(Periodic => [[3, 8], [.1], [.2, 1]],
                  RandomWalk => [collect(15:10:45)],
                  UniformLinear => [collect(4:4:16)],
                  :noise => [.0001, .0005])


@gen function covariance_prior(kernel_type, dot, dim)
    if kernel_type == Periodic
        kernel_args = [{(:amplitude, dot, dim)} ~ multinomial(param_dict[Periodic][1]),
                       {(:lengthscale, dot, dim)} ~ multinomial(param_dict[Periodic][2]),
                       {(:period, dot, dim)} ~ multinomial(param_dict[Periodic][3])]
    elseif kernel_type == UniformLinear
        kernel_args = [{(:covariance, dot, dim)} ~ multinomial(param_dict[UniformLinear][1])]
    elseif kernel_type == RandomWalk
        kernel_args = [{(:variance, dot, dim)} ~ multinomial(param_dict[RandomWalk][1])]
    end
    return kernel_type(kernel_args...)
end

@dist gamma_bounded_below(shape, scale, bound) = gamma(shape, scale) + bound

@dist multinomial(possibilities) = possibilities[uniform_discrete(1, length(possibilities))]

macro datatype(str); :($(Symbol(str))); end





                                          

