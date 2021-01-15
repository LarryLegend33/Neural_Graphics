using Makie
using AbstractPlotting
using MakieLayout


function animate_spike_train(spike_times::Array{Float64, 2})
    num_frames = 500
    res = 500
    white = RGBf0(255,255,255)
    black = RGBf0(0,0,0)
    gray = RGBf0(100, 100, 100)
    scene, layout = layoutscene(backgroundcolor=black, resolution=(res, res))
    axes = LAxis(scene, backgroundcolor=black, ylabelcolor=white, xlabelcolor=white,
                 xticklabelcolor=white, yticklabelcolor=black, 
                 xtickcolor=white, ytickcolor=black, xgridcolor=black, ygridcolor=black,
                 xticklabelrotation = pi/2,  xticklabelalign = (:top, :top))
    layout[1, 1] = axes
    f(time, coords, neuron_id) = vcat([[(x, float(neuron_id)), (x, float(neuron_id+1))] for x in time .- coords[:, neuron_id]]...)
    time_node = Node(1)
    for neuron in 1:size(spike_times)[2]
        linesegments!(axes, lift(t->f(t, spike_times, neuron), time_node), color=white)
    end
    axes.ylabel = "Spike Raster"
    axes.xlabel = "Time (ms)"
    limits!(axes, BBox(0, 200, 0, size(spike_times)[2] + 1))
    display(scene)
    for i in 1:num_frames
        time_node[] = i
        sleep(1/100)
    end
end
    
function draw_gibbs_circuit(n_integrators::Int,
                            n_poisson::Int)
    res = 600
    num_frames = 20
    colors = [i % 2 == 0 ? :white : :blue for i in 1:num_frames]
    f(t, c) = c[t]
    time_node = Node(1);
    scene = Scene(backgroundcolor=:black, resolution=(res+200, res))
    integrator_coords = [(x, 4) for x in 1:n_integrators]
    poisson_coords = [(x, 2) for x in 1:n_poisson]
    poisson_coords2 = [(x, 1) for x in 1:n_poisson]
    # first two args after scene are source of the vector. next two are deltas for the vector. 
    scatter!(scene, integrator_coords, color=lift(t-> f(t, colors), time_node), marker=:hexagon, markersize=1)
    scatter!(scene, vcat(poisson_coords, poisson_coords2), color=lift(t-> f(t, colors), time_node), marker=:circle, markersize=.3)

    
    text!(scene, string(v), position=(node_xs[v], node_ys[v]), align= (:center, :center),
          textsize=.2, color=:black, overdraw=true)
 #   arrows!(scene, 1:n_integrators, zeros(n_integrators), zeros(n_integrators), ones(n_integrators),
  #          linecolor=lift(t-> f(t, colors), time_node), marker=:vline)
    ylims!(scene, 0, 5)
#    xlims!(scene, 0, 5)
    display(scene)
    for i in 1:num_frames
        time_node[] = i
        sleep(1/2)
    end
end    
