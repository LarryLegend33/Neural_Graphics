using AbstractPlotting
using AbstractPlotting.MakieLayout
using GLMakie

# Want to have the template
# And also the currently inferred velocity.
# And a graph that describes how close the proposal is to the correctc answer
# Also a scene graph estimate with probabilities.
# But for now two side by side renderings is good. 


#= this is a list of 7 dot locations ranging from -25/2 to 25/2 in both axes. 
the idea is to create 1000 instances of 7 dot coordinates =#
res = 1000
outer_padding = 0
scene, layout = layoutscene(outer_padding, 
                        resolution=(2*res, res), 
                        backgroundcolor=RGBf0(0,0,0))

dot1 = [(rand(range(0, stop=300)), rand(range(0, stop=300))) for i in range(1, stop=1000)]

f(t, coords) = (coords[t][1], coords[t][2])

ncols = 2
nrows = 1
# create a grid of LAxis objects
axes = [LAxis(scene) for i in 1:nrows, j in 1:ncols]
# and place them into the layout
layout[1:nrows, 1:ncols] = axes

time_node = Node(0.0)

scatter!(axes[1, 1], lift(t -> f.(t, dot1), time_node))
#scatter!(axes[1, 2], [coord[t][1]], [coord[1][2]], markersize = 20)

#s = scene[end] # last plot in scene
# record(scene, "output.mp4", r) do m
#     s[1] = m[:, 1]
#     s[2] = m[:, 2]

record(scene, "output.mp4", range(1, stop=1000), framerate=20) do i
    time_node[] = i
end




