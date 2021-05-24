using Gen
using StatsBase
using Serialization
using GLMakie
using CairoMakie

# UTILS:
# onehot vector for `x` with length `length(dom)`,
# with `x` truncated to domain
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
### Model
XDOMAIN = collect(1:10)

@gen function object_motion_step(T::Int64)
    x = 5
    xs = []
    for t in 1:T
        x = {t => :x} ~ categorical(maybe_one_off(x, 0.8, XDOMAIN))
        obs = {t => :obs} ~ categorical(maybe_one_off(x, 0.5, XDOMAIN))
        push!(xs, x)
    end
    return xs
end
### Proposal

@gen function step_proposal(t_old, obs, t)
    x = {t => :x} ~ categorical(maybe_one_off(obs, 0.5, XDOMAIN))
end


gt = [(4, 3),
      (3, 3),
      (3, 3),
      (4, 5),
      (4, 3),
      (4, 4),
      (3, 2),
      (4, 4),
      (3, 2),
      (2, 2),
      (2, 2),
      (2, 2),
      (3, 3),
      (2, 2),
      (3, 3),
      (4, 4),
      (3, 3),
      (4, 5),
      (4, 4),
      (3, 4),
      (2, 2)]


gtv = [(2, 4, 3),
       (3, 5, 6),
       (2, 6, 5),
       (2, 7, 7),
       (1, 7, 7),
       (1, 6, 7),
       (2, 5, 4),
       (2, 5, 4),
       (3, 4, 5),
       (3, 6, 6),
       (2, 6, 6),
       (2, 6, 6),
       (3, 6, 6),
       (3, 7, 7),
       (3, 8, 7),
       (2, 8, 7),
       (3, 7, 7),
       (3, 8, 9),
       (3, 10, 10),
       (3, 10, 10),
       (3, 10, 10)]


neural_position_states = [[4, 4, 4, 4, 4, 4, 4, 4, 4, 4], 
                          [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                          [2, 4, 2, 4, 4, 4, 2, 2, 2, 2], 
                          [4, 5, 4, 4, 5, 4, 4, 4, 4, 4], 
                          [4, 3, 4, 3, 3, 4, 4, 4, 4, 3],
                          [4, 5, 5, 4, 3, 3, 5, 5, 5, 5],
                          [2, 3, 2, 2, 3, 3, 3, 3, 3, 3],
                          [4, 4, 4, 3, 4, 4, 4, 4, 4, 3],
                          [3, 3, 2, 3, 2, 2, 3, 2, 3, 3],
                          [2, 2, 2, 2, 3, 3, 3, 2, 2, 2],
                          [3, 3, 3, 3, 3, 3, 2, 3, 3, 2],
                          [2, 2, 2, 2, 2, 3, 2, 2, 2, 2],
                          [3, 3, 2, 3, 3, 3, 3, 3, 4, 3],
                          [2, 2, 3, 2, 2, 2, 2, 2, 2, 2],
                          [3, 2, 3, 3, 3, 3, 3, 3, 3, 2],
                          [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]


@gen function object_motion_step_velocity(T::Int)
    vel = 2
    x = 5
    xs = []
    for t in 1:T
        vel = {t => :vel} ~ categorical(
            vel == 1 ? [.7, .3, .0] : # vel=-1
                vel== 2 ? [.3, .4, .3] : # vel= 0
                [.0, .3, .7])   # vel = 1
        x = {t => :x} ~ categorical(maybe_one_off(x + vel-2, 0.4, XDOMAIN))
        obs = {t => :obs} ~ categorical(maybe_one_off(x, 0.5, XDOMAIN))
        push!(xs, x)
    end
    return xs
end
### Proposal
@gen function step_proposal_velocity(tr_old, obs, t)
    x = {t => :x } ~ categorical(maybe_one_off(obs, 0.5, XDOMAIN))
    if t > 1
        xminus1 = tr_old[t-1 => :x]
    else
        xminus1 = 5
    end
    vel = {t => :vel} ~ categorical(maybe_one_off(x-xminus1 + 2, 0.5, 1:3))
end


# note without resampling state.trace[1] does just extend its timeseries.
# particle filter is working. 

function linepos_particle_filter(num_particles::Int, num_samples::Int, observations::Array{Int64}, gen_function::DynamicDSLFunction{Any}, proposal)
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
    location_matrix = zeros(times, length(XDOMAIN))
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

function heatmap_pf_results(state, latent_v::Symbol)
    times = length(get_retval(state.traces[1]))
    observations = [state.traces[1][t => :obs] for t in 1:times]
    location_matrix = zeros(times, length(XDOMAIN))
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

function heatmap_neural(neural_states, observations)
    times = length(neural_states)
    observations = observations[1:times]
    n_neurons = length(neural_states[1])
    location_matrix = zeros(times, length(XDOMAIN))
    for (t, state) in enumerate(neural_states)
        for n in 1:n_neurons
            location_matrix[t, state[n]] += 1
        end
    end
    fig = Figure(resolution=(1000,1000))
    ax = fig[1, 1] = Axis(fig)
    hm = heatmap!(ax, location_matrix)
    cbar = fig[1, 2] = Colorbar(fig, hm, label="N Particles")
    scatter!(ax, [t-.5 for t in 1:times], [o-.5 for o in observations], color=:magenta)
    display(fig)
end                            
           
            
            
        
                                 
