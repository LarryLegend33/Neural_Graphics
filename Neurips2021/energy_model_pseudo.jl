using Gen
using StatsBase
using Serialization
using GLMakie
using CairoMakie
using Distributions
using BSON: @save, @load

onehot(x, dom) =
    x < first(dom) ? onehot(first(dom), dom) :
    x > last(dom)  ? onehot(last(dom), dom)  :
                 [i == x ? 1. : 0. for i in dom]
# prob vector to sample a value in `dom` which is 1 off
# from `idx` with probability `prob`, and `idx` otherwise
maybe_one_off(idx, prob, dom) =
    (1 - prob) * onehot(idx, dom) +
    prob/2 * onehot(idx - 1, dom) +
    prob/2 * onehot(idx + 1, dom)
maybe_one_or_two_off(idx, prob, dom) = 
    (1 - prob) * onehot(idx, dom) +
    prob/3 * onehot(idx - 1, dom) +
    prob/3 * onehot(idx + 1, dom) +
    prob/6 * onehot(idx - 2, dom) +
    prob/6 * onehot(idx + 2, dom)


normalize(v) = v / sum(v)
discretized_gaussian(mean, std, dom) = normalize([
    cdf(Normal(mean, std), i + .5) - cdf(Normal(mean, std), i - .5) for i in dom
])

@dist LabeledCategorical(labels, probs) = labels[categorical(probs)]

Xs = collect(1:20)
HOME = 10
Vels = collect(-3:3)
Energies = collect(1:10)
XInit = 10
EInit = 5
VThatLeadToXInit = 0

# when its 6 or 7 away, its likely to stop. 

moving_away_from_home(x, v) = sign(v) == sign(x - HOME)
dist_from_home(x) = abs(x - HOME)

τ_far() = 10
τ_tired() = 10
prior_p_stop_tired(e_prev) = exp(-e_prev/ τ_far())
prior_p_stop_far(x, v) = moving_away_from_home(x, v) ? 1-exp(-dist_from_home(x)/τ_far()) : 0.
prop_p_stop_far(is_stopped, v, x) = !is_stopped ? 0. : moving_away_from_home(x, v) ? 0.5 : 0.
prop_p_stop_tired(is_stopped, already_stopped, e_prev) = !is_stopped ? 0. : already_stopped ? prior_p_stop_tired(e_prev) : .6

expected_e(e, v) = e + (abs(v) > 0 ? -abs(v) : Energies[end] / 5)

# e drops if abs(v) > 0 proportionally to abs(v). if the previous
# move was a rest, you get 2 energies back. probably too much of an energy hit
# if abs(v) can be 4.



""" Pseudomarginalized Dist Definition """

struct PseudoMarginalizedDist{R} <: Gen.Distribution{R}
    model::Gen.GenerativeFunction{R}
    proposal::Gen.GenerativeFunction
    ret_trace_addr
    n_particles::Int
end

Gen.random(d::PseudoMarginalizedDist, args...) =
    get_retval(simulate(d.model, args))

function Gen.logpdf(d::PseudoMarginalizedDist, val, args...)
    weight_sum = 0.
    for _=1:d.n_particles
        proposed_choices, proposed_score = propose(d.proposal, (val, args...))
        assessed_score, v1 = assess(
            d.model, args,
            merge(choicemap((d.ret_trace_addr, val)), proposed_choices)
        )
        @assert v1 == val "val = $val, v1 = $v1"
        weight_sum += exp(assessed_score - proposed_score)
    end
    return log(weight_sum) - log(d.n_particles)
end


@gen function vel_model(e_curr, v_prev, x_curr)
    stop_bc_tired = { :stop_tired } ~ bernoulli(prior_p_stop_tired(e_curr))
    # 1 away, 4.8% chance of stopping. 10 away, 40%
    stop_bc_far_from_home = { :stop_far } ~ bernoulli(prior_p_stop_far(x_curr, v_prev))
    stop = stop_bc_tired || stop_bc_far_from_home
    v = { :v } ~ LabeledCategorical(Vels, 
        stop ? onehot(0, Vels) :
            maybe_one_or_two_off(v_prev, .8, Vels))
    return v
end

# need to marginalize out these two variables so we have to consider what we already know
# would really like to sum over all possible values, but instead are going to sample tons of possibilities. 

# This is given that you have already OBSERVED current v. if its zero, much more likely to propose that you stopped.
# This part of the proposal occurs before you move on to proposing x and e values. 
@gen function vel_auxiliary_proposal(v, e_curr, v_prev, x_curr)
    stop_because_far = { :stop_far } ~ bernoulli(prop_p_stop_far(v == 0, v_prev, x_curr))
    stop_because_tired = { :stop_tired } ~ bernoulli(prop_p_stop_tired(v == 0, stop_because_far, e_curr))
end

vel_dist = PseudoMarginalizedDist(
    vel_model,
    vel_auxiliary_proposal,
    :v,
    2 # TODO: tune NParticles
)


@gen function position_step_model(v_prev, e_curr, x_curr)
    # make a step depending on your energy and x location, and your previous velocity
    v = { :v } ~ vel_dist(e_curr, v_prev, x_curr)
    # this is your next energy state after the first velocity move.
    e = { :e } ~ categorical(maybe_one_off(expected_e(e_curr, v), .5, Energies))
    x = { :x } ~ categorical(maybe_one_off(x_curr + v, .2, Xs))
    # Play with std
    obs = { :obs } ~ categorical(discretized_gaussian(x, 2.0, Xs))
    return (v, e, x, obs)
end


# here you have observed an X value. propose that the latent true value is near it.
# propose that the velocity that lead to X is x-x_previous. propose that the energy after
# the V step is based on e_prev and the V step. 

@gen function step_proposal(tr_old, obs, t)
    #    x = { t => :x } ~ categorical(discretized_gaussian(obs, 4.0, Xs))
    x = { t => :x } ~ categorical(maybe_one_off(obs, .4, Xs))
    if t > 1
        x_prev = tr_old[t-1 => :x]
        e_prev = tr_old[t-1 => :e]
        v_prev = tr_old[t-1 => :v]
    else
        x_prev = XInit
        e_prev = EInit
        v_prev = VThatLeadToXInit
    end
    v = { t => :v } ~ categorical(maybe_one_off(x - x_prev, 0.5, Vels))
    { t => :e } ~ categorical(maybe_one_off(expected_e(e_prev, v), .5, Energies))
end

@gen function move_for_time(T::Int)
    x = XInit
    v = VThatLeadToXInit
    # this should be implicit. your velocity before this was 0.
    # XInit and EInit are added to the plot as the first member of the array. 
    e = EInit 
    xs = []
    obss = []
    for t in 1:T
        (v, e, x, obs) = {t} ~ position_step_model(v, e, x)
        push!(xs, x)
        push!(obss, obs)
    end
    return (xs, obss)
end



function linepos_particle_filter(num_particles::Int, gt_trace::Trace, gen_function::DynamicDSLFunction{Any}, proposal)
    observations = get_retval(gt_trace)[2]
    obs1 = Gen.choicemap((1 => :obs, observations[1]))
    if proposal == ()
        state = Gen.initialize_particle_filter(gen_function, (1,), obs1, num_particles)
    else
        state = Gen.initialize_particle_filter(gen_function, (1,), obs1, proposal, (gt_trace, observations[1], 1), num_particles)
    end
    for t in 2:length(observations)
        Gen.maybe_resample!(state, ess_threshold=num_particles)
        obs = Gen.choicemap((t => :obs, observations[t]))
        if proposal == ()
            Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
        else
            Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs, proposal, (observations[t],t))
        end
        println([state.traces[1][tind => :x] for tind in 1:t])
        println([get_score(state.traces[i]) for i in 1:num_particles if isfinite(get_score(state.traces[i]))])
    end
    return state
end


# am doing full resampling. if not, you can use Gen.sample_unweighted_traces(state, num_samples)
# to get a sample set back. its kind of a misnomer -- it samples a categorical based on weights. 

function heatmap_pf_results(state, gt::Trace, latent_v::Symbol)
    true_x = get_retval(gt)[1]
    times = length(get_retval(state.traces[1])[2])
    observations = get_retval(state.traces[1])[2]
    # also plot the true x values
    location_matrix = zeros(length(Xs), times)
    for t in 1:times
        for tr in state.traces
            location_matrix[tr[t => latent_v], t] += 1
        end
    end
    fig = Figure(resolution=(1000,1000))
    ax = fig[1, 1] = Axis(fig)
    hm = heatmap!(ax, location_matrix)
    cbar = fig[1, 2] = Colorbar(fig, hm, label="N Particles")
    scatter!(ax, [o-.5 for o in observations], [t-.5 for t in 1:times], color=:magenta)
    scatter!(ax, [tx-.5 for tx in true_x], [t-.5 for t in 1:times], color=:white)
    vlines!(ax, HOME, color=:red)
    display(fig)
end


function x_trajectory_anim(trace)
    darkcyan = RGBAf0(0, 170, 170, 50) / 255
    times = length(get_retval(trace))
    xs = [trace[t=> :x] for t in 1:times]
    println(xs)
    obs = [trace[t=> :obs] for t in 1:times]
    t_node = Node(1)
    f(t) = [(x, 0) for x in xs[1:t]]
    fig = Figure()
    ax = fig[1,1] = Axis(fig)
    ylims!(ax, (-1, 1))
    xlims!(ax, (Xs[1]-1, Xs[end]+1))
    scatter!(ax, lift(t->f(t), t_node), color=darkcyan)
    display(fig)
    for t in 1:times
        t_node[] = t
        sleep(.5)
    end
end


function extract_and_plot_groundtruth(tr)
    times = length(get_retval(tr)[1])
    xs = vcat([XInit], [tr[t=> :x] for t in 1:times])
    vs = [tr[t=> :v] for t in 1:times]
    es = vcat([EInit], [tr[t=> :e] for t in 1:times])
    fig = Figure(resolution=(1500,2000 * times / length(Xs)))
    ax = fig[1,1] = Axis(fig, xgridvisible=false, ygridvisible=false)
    vlines!(ax, [HOME], color=:red)
    scatter!(ax, [xt for xt in zip(xs, 0:times)], colorrange = (1, Energies[end]), color=es, colormap= :thermal, marker=:rect, markersize=30)
    arrows!(ax, xs[1:end-1], 0:times-1, vs, ones(length(vs)))
    xlims!(ax, (Xs[1]-2, Xs[end]+2))
    ylims!(ax, (-1, times+1))
    display(fig)
    return vs, es
end


function save_run(tr, trace_file_id)
    trace_args, trace_choices = get_args(tr), get_choices(tr)
    @save string(trace_file_id) trace_args trace_choices
end
    
function load_run(trace_file_id)
    @load trace_file_id trace_args trace_choices
    (trace, w) = Gen.generate(move_for_time, trace_args, trace_choices)
    return trace
end

function generate_groundtruth(num_steps)
    tr = Gen.simulate(move_for_time, (num_steps, ))
    extract_and_plot_groundtruth(tr)
    return tr
end


# make sure you figure out why energy 1 is not constant. 

function proposal_test()
    run_out_for = 1
    replace_ind = 1
    tr = Gen.simulate(move_for_time, (1, ))
    obs = get_retval(tr)[2]
    tr, model_score_no_prop = Gen.generate(move_for_time, get_args(tr), get_choices(tr))
    # this comes out finite
    println(get_choices(tr))
    println(model_score_no_prop) 
    prop_choices, prop_score, retval = Gen.propose(step_proposal, (tr, get_retval(tr)[2][replace_ind], replace_ind))
    # this also comes out finite. all that's happening here is we are proposing new values for the 5th x, v, and e. 
    println(prop_score)
    println(prop_choices)
#    pc_as_dynamic_choicemap = Gen.choicemap(
    updated_tr, update_score, retdiff, discard = Gen.update(tr, get_args(tr), (), prop_choices)
    # this is never finite even though the trace is updated correctly with the proposal values. each [5 => latentv] enters the trace,
    # but the score comes back Inf regardless. 
    println(update_score)
    model_after_prop_tr, model_score_after_prop = Gen.generate(move_for_time, get_args(tr), prop_choices)
    println(model_score_after_prop)
    return prop_choices, get_choices(updated_tr)
end
   
