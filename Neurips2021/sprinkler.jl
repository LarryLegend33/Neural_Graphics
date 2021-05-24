using Gen
using StatsBase
using Serialization
using GLMakie
using CairoMakie


""" Static functions for Wet Grass Bayes Net """ 

@gen (static) function iswet(in::Nothing)
    raining ~ bernoulli(in === nothing ? 0.3 : 0.3)
    sprinkler ~ bernoulli(in === nothing ? 0.3 : 0.3)
    grasswet ~ bernoulli(raining || sprinkler ? 0.9 : 0.1)
    return grasswet
end

@gen (static) function raining_proposal(trace)
    raining ~ bernoulli(trace[:raining] ? 0.5 : 0.5)
    return raining
end

@gen (static) function sprinkler_proposal(trace)
    sprinkler ~ bernoulli(trace[:sprinkler] ? 0.5 : 0.5)
    return sprinkler
end

@load_generated_functions()

function run_unblocked_mh(initial_trace, iters)
    tr = initial_trace
    traces = []
    for _=1:iters
        tr, _ = Gen.mh(tr, raining_proposal, ())
        push!(traces, tr)
        tr, _ = Gen.mh(tr, sprinkler_proposal, ())
        push!(traces, tr)
    end
    return traces
end


@gen (static) function smart_block_proposal(trace)
    raining ~ bernoulli(trace[:grasswet] ? 0.55 : 0.2)
    sprinkler ~ bernoulli(trace[:grasswet] && !raining ? 0.9 : 0.3)
end

@load_generated_functions()

function run_block_mh(initial_tr, iters)
    tr = initial_tr
    traces = []
    for i=1:iters
        tr, _ = Gen.mh(tr, smart_block_proposal, ())
        push!(traces, tr)
    end
    return traces
end


function run_all_mh()
    initial_trace, _ = generate(iswet, (nothing,), choicemap((:grasswet, true), (:sprinkler, true), (:raining, true)))
    # run block MH on the same initial trace from above.
    b_traces = run_block_mh(initial_trace, 50)
    ub_traces = run_unblocked_mh(initial_trace, 50)
    # load results from neural SMC
    neural_smc_block = deserialize("snn_blocked_mh_states.jls")
    neural_smc_unblock = deserialize("snn_unblocked_mh_states.jls")
    return [b_traces, ub_traces, neural_smc_block, neural_smc_unblock]
end

function make_sprinkler_plot(gen_results, neural_results)
    skyblue = RGBf0(154, 203, 255) / 255
    darkcyan = RGBf0(0, 170, 170) / 255
    lightgreen = RGBf0(144, 238, 144) / 255
    magenta = RGBf0(255,128,255) / 255
    dark_green = RGBf0(34,139,34) / 255
    dark_blue = RGBf0(0, 60, 120) / 255
    fig = Figure(resolution=(2000, 1500))
    p_results = [gen_results, neural_results]
    p_symbols = [:grasswet, :sprinkler, :raining]
    p_colors = [lightgreen, magenta, darkcyan]
    for c in 1:length(p_results) for r in 1:length(p_symbols)
        ax_title = String(p_symbols[r])
        ax = fig[r, c] = Axis(fig, xgridvisible=false, ygridvisible=false, title=ax_title)
        println(ax_title)
        lines!(ax, [v[p_symbols[r]] for v in p_results[c]], color=p_colors[r], linewidth=2)
        ylims!(ax, (-.2, 1.2))
    end
    end
    display(fig)
    save_figure(fig)
end


function save_figure(fig)
    CairoMakie.activate!()
    save("output.svg", fig)
    GLMakie.activate!()
end




# util to count the number of times each assignment to `(raining, sprinkler)`  was visited,
# given `states` as a vector of `(raining, sprinkler)` pairs
# counts(states) = Dict(
#     (true, true) => length(filter(states) do (r, s); r && s; end),
#     (false, true) => length(filter(states) do (r, s); !r && s; end),
#     (true, false) => length(filter(states) do (r, s); r && !s; end),
#     (false, false) => length(filter(states) do (r, s); !r && !s; end))
