using DataFrames, RDatasets
using GLMakie
using LightGraphs
using MetaGraphs


# Want to have the template
# And also the currently inferred velocity.
# And a graph that describes how close the proposal is to the correctc answer
# Also a scene graph estimate with probabilities i.e. Austin's pres. 
# But for now two side by side renderings is good. 

"""WORKS AT BASICALLY FULL SPEED AT 1000 RESOLUTION"""

##eventually this is going to take a metagraph and plot the
##dot positions of each graph. 

# res = 1000
# outer_padding = 0
# num_updates = 180
# scene, layout = layoutscene(outer_padding,
#                             resolution = (2*res, res), 
#                             backgroundcolor=RGBf0(0, 0, 0))








# dot1 = [(rand(range(0, stop=1000)), rand(range(0, stop=1000))) for i in range(1, stop=1000)]
# dot2 = [[(i, .5), (i, .7), (i, .9)] for i in range(0, stop=1, length=num_updates)]

# f(t, coords) = coords[t]

# ncols = 2
# nrows = 1
# # create a grid of LAxis objects
# axes = [LAxis(scene, backgroundcolor=RGBf0(0, 0, 0)) for i in 1:nrows, j in 1:ncols]
# layout[1:nrows, 1:ncols] = axes

# time_node = Node(1);

# # first get an array of 3 dots into scatter. 

# scatter!(axes[1], lift(t -> f(t, dot2), time_node), markersize=20px, color=RGBf0(255, 255, 255))
# limits!(axes[1], BBox(0, 1, 0, 1))
# scatter!(axes[2], lift(t -> f(t, dot2), time_node), markersize=20px, color=RGBf0(255, 255, 255))
# limits!(axes[2], BBox(0, 1, 0, 1))
# display(scene)



# for i in range(1, stop=num_updates)
#     time_node[] = i
#     sleep(1/60)
# end

# #dot2 = [(i, .8) for i in range(0, stop=1, length=num_updates)]

# for i in range(1, stop=num_updates)
#     time_node[] = i
#     sleep(1/60)
# end

# # OK so you want to put this in a recursive loop where the only thing you're changing is
# # the dot coordinates. this can even be the main line, where you simply update
# # dot coordinates with the model's choices. yes! that is how you should do it.
# # write a function that takes graphs and gives back 3x500 arrays. keep left the actual answer,
# # right plot is the models' choices. dots 1 will be first draw, dots 2 will be models guesses.
# # lower left will be scene graph. lower right will be successive likelihoods.

# # Think about: when you plot the graph, do you want the current inferred
# # graph? or a probability of graphs as you go through the inference program.
# # the tricky thing about this is that there's going to be 3 seconds per visualization
# # there's basically no time for comparison of coords if you don't show it. 



# """ Adding a recording step significantly slows down onscreen animation, but movie is fine """
# # record(scene, "output.mp4", range(1, stop=num_updates), framerate=60, compression=0) do i
# #     time_node[] = i
# # end

# GLMakie.destroy!(GLMakie.global_gl_screen())




# Use a SimpleDigraph with 3 dots" g = SimpleDigraph(3)
# Add a vertex to none, dot 2 and/or dot 3: add_edge!(g, 1, 1)
# make a metagraph for the graph with mg = MetaGraph(g)
# Store info for each node with set_props!(mg, 3, Dict)



N = 1000
a = rand(1:2, N) # a discrete variable
b = rand(1:2, N) # a discrete variable
x = randn(N) # a continuous variable
y = @. x * a + 0.8*randn() # a continuous variable
z = x .+ y # a continuous variable
@substep

scatter(x, y, markersize = 0.2)
@substep

scatter(Group(a), x, y, markersize = 0.2)
@substep

scatter(Group(a), x, y, color = [:black, :red], markersize = 0.2)
@substep

scatter(Group(marker = a), x, y, markersize = 0.2)
@substep

scatter(Group(marker = a, color = b), x, y, markersize = 0.2)
@substep

scatter(Group(marker = a), Style(color = z), x, y)
@substep

scatter(Group(color = a), x, y, Style(markersize = z ./ 10))
@substep

plot(linear, x, y)
@substep

plot(linear, Group(a), x, y)
@substep

scatter(Group(a), x, y, markersize = 0.2)
plot!(linear, Group(a), x, y)
@substep

plot(linear, Group(linestyle = a), x, y)
@substep

N = 200
x = 10 .* rand(N)
a = rand(1:2, N)
y = sin.(x) .+ 0.5 .* rand(N) .+ cos.(x) .* a
@substep

scatter(Group(a), x, y)
plot!(smooth, Group(a), x, y)
@substep

plot(histogram, y)
@substep

plot(histogram, x, y)
@substep

plot(histogram(nbins = 30), x, y)
@substep

wireframe(histogram(nbins = 30), x, y)
@substep

iris = RDatasets.dataset("datasets", "iris")
scatter(Data(iris), Group(:Species), :SepalLength, :SepalWidth)
@substep
# use Position.stack to signal that you want bars stacked vertically rather than superimposed
plot(Position.stack, histogram, Data(iris), Group(:Species), :SepalLength)
@substep

wireframe(
    density(trim=true),
    Data(iris), Group(:Species), :SepalLength, :SepalWidth,
    transparency = true, linewidth = 0.1
)
@substep

scatter(
    Data(iris),
    Group(marker = :Species, color = bycolumn),
    :SepalLength, (:PetalLength, :PetalWidth)
)
@substep

barplot(["hi", "ima", "string"], rand(3))
@substep
