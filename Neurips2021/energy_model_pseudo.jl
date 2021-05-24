using Gen
using StatsBase
using Serialization
using GLMakie
using CairoMakie
using Distributions

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
Xs = collect(1:40)
HOME = 20
Vels = collect(-4:4)
Energies = collect(1:30)
XInit = 10
EInit = 10

moving_away_from_home(x_tminus1, v_tminus1) = sign(v_tminus1) == sign(x_tminus1 - HOME)
dist_from_home(x_tminus1) = abs(x_tminus1 - HOME)
τ_far() = 10

# e drops if abs(v) > 0 proportionally to abs(v). if the previous
# move was a rest, you get 2 energies back. probably too much of an energy hit
# if abs(v) can be 4.
expected_e(e_prev, v) = e_prev + (abs(v) > 0 ? -abs(v) : 2)


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


@gen function vel_model(e_tminus1, v_tminus1, x_tminus1)
    stop_bc_tired = { :stop_tired } ~ bernoulli(exp(-e_tminus1))
    # 1 away, 4.8% chance of stopping. 10 away, 40%
    stop_bc_far_from_home = { :stop_far } ~ bernoulli(
        moving_away_from_home(x_tminus1, v_tminus1) ?
            1-exp(-dist_from_home(x_tminus1)/τ_far()) :
            0.)
    stop = stop_bc_tired || stop_bc_far_from_home
    v = { :v } ~ LabeledCategorical(Vels, 
        stop ? onehot(0, Vels) :
            maybe_one_or_two_off(v_tminus1, .8, Vels))
    return v
end

# need to marginalize out these two variables so we have to consider what we already know
# would really like to sum over all possible values, but instead are going to sample tons of possibilities, 
# which basically proposes the most important values contributing to the sum 
@gen function vel_auxiliary_proposal(v, e_tminus1, v_tminus1, x_tminus1)
    if v != 0
        { :stop_tired } ~ bernoulli(0)
        { :stop_far } ~ bernoulli(0)
        return
    end
    p_stopped_tired = exp(-e_tminus1)
    p_stopped_far = moving_away_from_home(x_tminus1, v_tminus1) ? 1-exp(-dist_from_home(x_tminus1)/τ_far()) : 0.
    p_neither_stop_true = (1 - p_stopped_tired)*(1 - p_stopped_far)
    p_stopvar_true = 1 - p_neither_stop_true
    # p (v = 0   ;  did not stop, v_tminus1)   (Prob v = 0 from being 1 or 2 off)
    # TODO: remove hardcoded prob 0.5
    p_off = maybe_one_or_two_off(0, 0.5, Vels)[v_tminus1 - first(Vels) + 1]
    approx_marginal_p_stop = p_stopvar_true / (p_stopvar_true + p_off)
    approx_p_tired_given_stop = p_stopped_tired / (p_stopped_tired + p_stopped_far)
    stop_tired = { :stop_tired } ~ bernoulli(approx_marginal_p_stop * approx_p_tired_given_stop)
    { :stop_far } ~ bernoulli(stop_tired ? p_stopped_far : approx_marginal_p_stop)
end

vel_dist = PseudoMarginalizedDist(
    vel_model,
    vel_auxiliary_proposal,
    :v,
    10 # TODO: tune NParticles
)

@gen function position_step_model(v_prev, e_prev, x_prev)
    v = { :v } ~ vel_dist(e_prev, v_prev, x_prev)
    e = { :e } ~ categorical(maybe_one_off(expected_e(e_prev, v), .5, Energies))
    x = { :x } ~ categorical(maybe_one_off(x_prev + v, .8, Xs))
    # Play with std
    obs = { :obs } ~ categorical(discretized_gaussian(x, 1.0, Xs))
    return (v, e, x, obs)
end


@gen function step_proposal(tr_old, obs, t)
    x = { t => :x } ~ categorical(discretized_gaussian(obs, 2.0, Xs))
    if t > 1
        x_prev = tr_old[t-1 => :x]
        e_prev = tr_old[t-1 => :e]
    else
        x_prev = XInit
        e_prev = EInit
    end
    v = { t => :v } ~ categorical(maybe_one_or_two_off(x - x_prev, 0.5, Vels))
    { t => :e } ~ categorical(maybe_one_off(expected_e(e_prev, v), .5, Energies))
end

@gen function move_for_time(T::Int)
    x = XInit
    v = 3
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

# @gen function proposal_over_time(T)
#     for t in 1:T
#         {t} ~ step_proposal(..)
#     end
# end

function linepos_particle_filter(num_particles::Int, num_samples::Int, observations::Vector{Any}, gen_function::DynamicDSLFunction{Any}, proposal)
    state = Gen.initialize_particle_filter(gen_function, (0,), Gen.choicemap(), num_particles)
    for t in 1:length(observations)
        Gen.maybe_resample!(state, ess_threshold=num_particles) #/2)
        obs = Gen.choicemap((t => :obs, observations[t]))
        if proposal == ()
            Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
        else
            Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs, proposal, (observations[t],t))
        end
        println([state.traces[1][tind => :x] for tind in 1:t])
    end
    return state
end



function heatmap_pf_results(state, latent_v::Symbol)
    times = length(get_retval(state.traces[1])[1])
    observations = get_retval(state.traces[1])[2]
    true_x = get_retval(state.traces[1])[1]
    # also plot the true x values
    location_matrix = zeros(times, length(Xs))
    for t in 1:times
        for tr in state.traces
            location_matrix[t, tr[t => latent_v]] += 1
        end
    end
    fig = Figure(resolution=(1000,1000))
    ax = fig[1, 1] = Axis(fig)
    hm = heatmap!(ax, location_matrix)
    cbar = fig[1, 2] = Colorbar(fig, hm, label="N Particles")
    scatter!(ax, [t-.5 for t in 1:times], [o-.5 for o in observations], color=:magenta)
    scatter!(ax, [t-.5 for t in 1:times], [tx-.5 for tx in true_x], color=:white)
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


