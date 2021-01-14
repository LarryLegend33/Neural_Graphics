using Makie
using AbstractPlotting
using MakieLayout


function animate_spike_train(spike_times::Array{Float64, 1})
    num_frames = 500
    res = 500
    scene = Scene(backgroundcolor=:black, resolution=(res, res))
    f(t, c) = t .- spike_times
    time_node = Node(1)
    arrows!(scene, lift(t->f(t, spike_times), time_node), zeros(10), zeros(10), ones(10), marker=:vline, linecolor=:white)
    xlims!(scene, (0, 500))
    ylims!(scene, (0, 5))
    display(scene)
    for i in 1:num_frames
        time_node[] = i
        sleep(1/60)
    end
end
    
function draw_gibbs_circuit(n_integrators::Int,
                            n_poisson::Int)
    res = 600
    num_frames = 20
    colors = [i % 2 == 0 ? :white : :blue for i in 1:num_frames]
    f(t, c) = c[t]
    time_node = Node(1);
    scene = Scene(backgroundcolor=:black, resolution=(res, res))
    integrator_coords = [(x, 3) for x in 1:n_integrators]
    poisson_coords = [(x, 2) for x in 1:n_poisson]
    poisson_coords2 = [(x, 1) for x in 1:n_poisson]
    # first two args after scene are source of the vector. next two are deltas for the vector. 
    scatter!(scene, integrator_coords, color=lift(t-> f(t, colors), time_node), marker=:hexagon)
    scatter!(scene, vcat(poisson_coords, poisson_coords2), color=lift(t-> f(t, colors), time_node), marker=:circle)
    
    arrows!(scene, 1:n_integrators, zeros(n_integrators), zeros(n_integrators), ones(n_integrators),
            linecolor=lift(t-> f(t, colors), time_node), marker=:vline)
    ylims!(scene, 0, 5)
    display(scene)
    for i in 1:num_frames
        time_node[] = i
        sleep(1/2)
    end
end    
