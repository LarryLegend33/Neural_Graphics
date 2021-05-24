using Gen
using StatsBase
using Serialization
using GLMakie
using CairoMakie


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


@dist LabeledCategorical(labels, probs) = labels[categorical(probs)]

Xs = collect(1:40)
HOME = 20
Vels = collect(-4:4)
Energies = collect(1:30)


@gen function stopping_model(e_tminus1, v_tminus1, x_tminus1, t)

    stop_bc_tired = {t => :stop_tired} ~ bernoulli(exp(-e_tminus1))
    # 1 away, 4.8% chance of stopping. 10 away, 40%
    τ_far = 10
    dist_from_home = abs(x_tminus1 - HOME)
    moving_away_from_home = sign(v_tminus1) == sign(x_tminus1 - HOME)
    stop_bc_far_from_home = {t => :stop_far} ~ bernoulli(moving_away_from_home ? 1-exp(-dist_from_home/τ_far) : 0.)

    return stop_bc_tired || stop_bc_far_from_home
end

@gen function position_step_model(T::Int)

    e_restgain = 2
    x = 20
    v = 3
    e = 10
    xs = []
    for t in 1:T
        stop = {*} ~ stopping_model(e, v, x, t)
        v = {t => :v} ~ LabeledCategorical(Vels, 
            stop ? onehot(0, Vels) :
                maybe_one_or_two_off(v, .8, Vels))
        x = {t => :x} ~ categorical(maybe_one_off(x + v, .8, Xs))
        push!(xs, x)

        # make a more complicated noise model
        obs = {t => :obs} ~ categorical(maybe_one_off(x, 0.5, Xs))
        # e drops if abs(v) > 0 proportionally to abs(v). if the previous
        # move was a rest, you get 2 energies back. probably too much of an energy hit
        # if abs(v) can be 4.
        expected_e = e + (abs(v) > 0 ? -abs(v) : e_restgain)
        e = {t => :e} ~ categorical(maybe_one_off(expected_e, .5, Energies))
    end
    return xs
end


function linepos_particle_filter(num_particles::Int, num_samples::Int, observations::Vector{Any}, gen_function::DynamicDSLFunction{Any}, proposal)
    state = Gen.initialize_particle_filter(gen_function, (0,), Gen.choicemap(), num_particles)
    for t in 1:length(observations)

#        Gen.maybe_resample!(state, ess_threshold=num_particles/2)
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
    times = length(get_retval(state.traces[1]))
    observations = [state.traces[1][t => :obs] for t in 1:times]
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
               


#=
P(eₜ, xₜ, vₜ ; eₜ₋₁, vₜ₋₁, xₜ₋₁)
P(vₜ ; eₜ₋₁, vₜ₋₁, xₜ₋₁) * P(xₜ ; vₜ), xₜ₋₁ * P()

p s.t. E[p] = P(vₜ ; eₜ₋₁, vₜ₋₁, xₜ₋₁)


Trace T = (stop_because_tired, stop_because_far_from_home, vₜ)
P(T ; eₜ₋₁, vₜ₋₁, xₜ₋₁)

IS:
N times:
Propose T_i  ~ Q( ; eₜ₋₁, vₜ₋₁, xₜ₋₁,    vₜ)
Score w_i = P(T ; eₜ₋₁, vₜ₋₁, xₜ₋₁) / Q(T ; ...)

1/N × ∑{w_i} ≈ P(vₜ ; eₜ₋₁, vₜ₋₁, xₜ₋₁)
as N --> infinity


P(T, vₜ; xₜ₋₁)
Q(T ; xₜ₋₁, vₜ)

E_{T~Q}[W_i]
∑_T{     w_i     Q(T ; xₜ₋₁, vₜ)  }

∑_T{     P(T, vₜ ;  xₜ₋₁) / Q(T ; xₜ₋₁, vₜ)     Q(T ; xₜ₋₁, vₜ)  }
∑_T{     P(T, vₜ ;  xₜ₋₁)  }
P(vₜ ;  xₜ₋₁)
=#



# @gen function loop()
#     state1 ~ initial_state_model() # write this
#     for __
#         {t} ~ step_model()
#     end
# end


#=



=#




# Observation model:










