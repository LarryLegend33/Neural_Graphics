using GLMakie
using Gen
using GenGridEnumeration
using OrderedCollections
using Random
using Statistics
using StatsBase
using GeometryBasics
import Images: load, Gray
using ImageFiltering
knl = ImageFiltering.Kernel


# Place camera on Y axis


greet() = Hello!


# note there are residual types in Makie from GeometryTypes, which is
# deprecated in favor of GeometryBasics

# Want to eventually do a type definition for shapes. will have vertices and faces as fields,
# will redefine GLMakie.mesh and wireframe to take Shape objects and render them.
# will be able to compose composite shapes using the kernel func type trees. also then
# assign translations and rotations using the function composition from pointlight. 



# gaussian blur in the model before running the likelihood model
# class of likelihood models that are better than pixel space
# essentially based on edge detection -- take image and produce a list of
# points w/ a coordinate along edges. points dont have IDs. rendering is doing a job. run edge detector that returns points along hte line. consider two sets of points to compare. chamfer distance between the points. what is likelihood of these points wrt the ones i generated? given the rendered points, how do you simulate data? one way of doing it is sampling N points you will generate. randomly pick an input point, add noise. mixture of gaussians of points. gives low weight to far away points. NxM input and output points in terms of computaiton. could compute w/ KD trees w/ pointcloud. grey noise or salt and pepper binary noise. no guidance towards answer. unless its coarse . relative to bottom up neural net. chamfer distance on point clouds. depth image: if you've got one, how do you score a synthetic depth image vs. observed one -- in a better way than just comparing w/ gaussians. pixel wise robust. distribution on scales of the 3D model. each section of 3D model has a name and identity: wheel, hull, exhaust pipe. points in 3D space; you render those and get 2D projections.

# given distance and fit to the 2D projection, its one thing. given fit to the 2D projection and another distance its another thing. knowledge of the rendering process -- this is common sense. big or small. what happens inside the generative model vs. the raw sensor. then there's the rest of your high level stuff. depth image is feedforward. separate from everything going on with the RGB image. chamfer distance marginalizing over all correspondences. multi object tracking work -- much more in depth into that problem. 



# Currently coming up w/ the inverted explanation b/c of the way images and matricies are described.


# cube vs. pyramid -- probability
# mesh vs wireframe -- add in possibility of giving random images.
# make visualization w/ fixed axis after talking to ben
# pick an outlier shape.
# output binary. how small grid can be how small bins can be.
# mh inside ben's example.
# each pixel is a binary value. space of values has to be discrete.
# noise model -- bit flipping by box. boxfilter > 4, bernouli = num_filed / 9.
# map in dynamic DSL. try julia conv

rotation_bounds = collect(-.5:.05:.5)

@dist labeled_cat(labels, weights) = labels[categorical(weights)]

@dist uniform_discrete_floats(value_range) = value_range[uniform_discrete(1,length(value_range))]

function make_cube_mesh(sidelen::Float64)
    vertices = sidelen*Point{3, Float64}[
        (1/2, 1/2, 1/2),
        (1/2, -1/2, 1/2),
        (-1/2, -1/2, 1/2),
        (-1/2, 1/2, 1/2),
        (1/2, 1/2, -1/2),
        (1/2, -1/2, -1/2),
        (-1/2, -1/2, -1/2),
        (-1/2, 1/2, -1/2),
    ]
    faces = QuadFace{Cint}[
        1:4,
        5:8,
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 8, 4],
        [4, 8, 5, 1]]
    cube_mesh = GeometryBasics.Mesh(vertices, faces)
    return cube_mesh
end    


function make_tetrahedron_mesh(sidelen::Float64)
    vertices = sidelen*Point{3, Float64}[
        (sqrt(8/9), 0, -1/3),
        (-sqrt(2/9), sqrt(2/3), -1/3),
        (-sqrt(2/9), -sqrt(2/3), -1/3),
        (0, 0, 1)
    ]
    faces = TriangleFace{Int}[
        [1,2,3],
        [1,2,4],
        [2,3,4],
        [1,3,4]]
    tetrahedron_mesh = GeometryBasics.Mesh(vertices, faces)
    return tetrahedron_mesh
end

    
function enumeration_grid(input_trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    tr_w_constraints = make_2D_constraints(input_trace)
    g = UniformPointPushforwardGrid(tr_w_constraints, OrderedDict(
#        :shape_choice => DiscreteSingletons([:cube, :tetrahedron]),
 #       :side_length => DiscreteSingletons([1, 2]),
        :rot_x => DiscreteSingletons(rotation_bounds),
      #  :rot_y => DiscreteSingletons(collect(0:1:π)),
        :rot_z => DiscreteSingletons(rotation_bounds)))
    makie_plot_grid(g, :rot_x, :rot_z)
    println(input_trace[:rot_x])
    println(input_trace[:rot_z])
    return g
end

# notes: w/ 2.94, .08 groundtruths, switching the order of rot_x and rot_z in g = had an effect, as did
# switching the axis of makie_plot_grid (i.e. putting :rot_z first). w/ normal order g and plot_grid,
# got the inverted axis effect. 


function make_2D_constraints(input_trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    constraints = Gen.choicemap()
    constraints[:shape_choice] = input_trace[:shape_choice]
    set_submap!(constraints, :image_2D, get_submap(get_choices(input_trace), :image_2D))
    tr, w = Gen.generate(primitive_shapes, (), constraints)
    return tr
end

function grid_mh_inference(input_trace)
    constraints = make_2D_constraints(input_trace)
    (new_tr, did_accept, grid, I_chosen, p_accept) = GenGridEnumeration.grid_drift_mh(
        tr,
        OrderedDict(:rot_x => IntervalPartition(logit(loc=π/2, scale=.1), 30),
                    :rot_z => IntervalPartition(logit(loc=π/2, scale=.1), 30)), 
        OrderedDict{Symbol, DiscreteSingletons}())
    (did_accept, p_accept)
end

# can also get positions w/out going to mesh first if you want to. i.e. vertices = decompose(Point3, cube_prim)
# can construct a primitive as well w/ a set of vertices (i.e. 

#camera(mesh_axis.scene).eyeposition

sphere_prim = GeometryBasics.Sphere(Point3(0.0, 0.0, 0.0), 1) # origin, radius
cylinder_prim = Cylinder(Point3(0.0, 0.0, 0.0), Point3(0.0,0.0,1.0), 1.0) #(origin, normal vector, width)

function shape_wrap()
    trace = Gen.simulate(primitive_shapes, ())
    shape_wrap(trace)
end

function shape_wrap(trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    fig = Figure()
    mesh_fig, projected_grid, shape = get_retval(trace)
    save("test.png", projected_grid)
    display(mesh_fig)
    return trace, mesh_fig
end

function shape_wrap(rot_x::Float64, rot_z::Float64, shape::Symbol)
    constraints = Gen.choicemap()
    constraints[:shape_choice] = shape
    constraints[:rot_x] = rot_x
    constraints[:rot_z] = rot_z
    tr, w = Gen.generate(primitive_shapes, (), constraints)
    shape_wrap(tr)
end

# Note showing the reshaped image with GLMakie.image shows an inverted pic relative to the saved image. May have
# something to do with the inverted outcomes in inference

# Use Axis3D objects and Figure declarations. Axis3D just needs to take the figure as an arg. 


@gen function primitive_shapes()
    shape_type = { :shape_choice } ~ labeled_cat([:cube, :tetrahedron], [1/2, 1/2])
    #    side_length = { :side_length } ~ uniform_discrete(1,2)
    side_length = 1.0
    if shape_type == :cube
        shape = make_cube_mesh(convert(Float64, side_length))
    elseif shape_type == :tetrahedron
        shape = make_tetrahedron_mesh(convert(Float64, side_length))
    end
    rotation_x = { :rot_x } ~ uniform_discrete_floats(rotation_bounds)
    rotation_z = { :rot_z } ~ uniform_discrete_floats(rotation_bounds)
#    rotation_y = { :rot_y } ~ uniform(0, 0)
    axis3_vectors = [Vec(1.0, 0.0, 0.0),
                 #    Vec(0.0, 1.0, 0.0),
                     Vec(0.0, 0.0, 1.0)]
#    quat_rotations = [qrotation(v, r) for (v, r) in zip(
 #       axis3_vectors, [rotation_x, rotation_y, rotation_z])]
    quat_rotations = [qrotation(v, r) for (v, r) in zip(
        axis3_vectors, [rotation_x, rotation_z])]
    rotation_quaternion = reduce(*, quat_rotations)
    mesh_render = render_static_mesh(shape, rotation_quaternion, "wire")
    mesh_render.scene.center = false
    projected_grid = scene_to_matrix(mesh_render)
#    noisy_image = {*} ~ generate_blur(projected_grid, .01)
    noisy_image = {*} ~  generate_bitnoise(projected_grid, 3)
    return mesh_render, reshape(noisy_image, size(projected_grid)), shape
end


@gen function generate_blur(grid::Matrix{Float64}, noiselevel::Float64)
    blurred_grid = imfilter(grid, ImageFiltering.Kernel.gaussian(1))
    noisy_image = ({ :image_2D } ~ noisy_matrix(blurred_grid, 0.1))
    return noisy_image
end


@gen function bernoulli_noisegen(p::Float64)
  # i.e. if all 9 are black pixels, still have a .1 chance of turning it white
  baseline_noise = .05 
  pix ~ bernoulli(p+baseline_noise)
  return pix
end


@gen function generate_bitnoise(im_mat::Matrix{Float64}, filter_size::Int)
    # if all 9 are white pixels, still have a .1 chance of going black.
    # this will be offset by the baseline noise the other way. 
    baseline_noise = .9
    conv_filter = baseline_noise*ones(filter_size, filter_size) / (filter_size^2)
    image_2D = { :image_2D } ~ Gen.Map(bernoulli_noisegen)(imfilter(im_mat, conv_filter))
end



#function rot_candidates


@gen function tile_proposal(tr)
    intervals = [π/8, π/4, π/2]
    tile_x = { :tile_x } ~ uniform_discrete_floats(intervals)
    tile_z = { :tile_z } ~ uniform_discrete_floats(intervals)
    rotation_bounds_x = collect(rot_candidates(tr[:rot_x], π/2, tile_x))
    rotation_bounds_x = collect(rot_candidates(tr[:rot_x], π/2, tile_z))
    rot_x = { :rot_x } ~ categorical(tr[:rot_x]*rotation_bounds_x)
    rot_z = { :rot_z } ~ categorical(tr[:rot_z]*rotation_bounds_z)
end
    


@gen function rotation_proposal(tr)
    rot_x = { :rot_x } ~ uniform_discrete_floats(calculate_rot_bounds(tr[:rot_x]))
    rot_z = { :rot_z } ~ uniform_discrete_floats(calculate_rot_bounds(tr[:rot_z]))
end


function shape_mh_update(tr_populated, amnt_computation)
    mh_traces = []
    accepted_list = []
    tr = make_2D_constraints(tr_populated)
    for i in 1:amnt_computation
       # (tr, accepted) = Gen.mh(tr, tile_proposal, ())
#        (tr, accepted) = Gen.mh(tr, rotation_proposal, ())
        (tr, accepted) = Gen.mh(tr, select(:rot_x, :rot_z))
        push!(mh_traces, tr)
        push!(accepted_list, accepted)
    end
    rot_mat = zeros(length(rotation_bounds), length(rotation_bounds))
    for (xi, x) in enumerate(rotation_bounds)
        for (zi, z) in enumerate(rotation_bounds)
            [rot_mat[xi,zi] += 1 for t in mh_traces if (x-.1 < t[:rot_x] < x+.1) && (z-.1 < t[:rot_z] < z+.1)]
        end
    end
    f, a = heatmap(rotation_bounds, rotation_bounds, rot_mat)
    display(f)
    return mh_traces, accepted_list
end



function animate_mh_chain(mh_traces)
    darkcyan = RGBAf0(0, 170, 170, 50) / 255
    tr_rot_x_z = [(tr[:rot_x], tr[:rot_z]) for tr in mh_traces]
    f(t) = tr_rot_x_z[1:t]
    f_color(t) = RGBAf0(t / length(mh_traces), t / length(mh_traces), 0, .2)
    time_node = Node(1)
    fig = Figure()
    ax = fig[1,1] = Axis(fig) 
    scatter!(ax, lift(t -> f(t), time_node), color=lift(t -> f_color(t), time_node))
    xlims!(ax, (-.6, .6))
    ylims!(ax, (-.6, .6))
    display(fig)
    for (i, tr) in enumerate(mh_traces)
        time_node[] = i
        sleep(.01)
    end
end

    

# 

function render_static_mesh(shape, rotation::Quaternion{Float64}, mesh_or_wire::String)
    white = RGBAf0(255, 255, 255, 0.0)
    res = 50
    mesh_fig = Figure(resolution=(res, res), figure_padding=-50)
    limval = 2.0
    lim = (-limval, limval, -limval, limval, -limval, limval)
    # note perspectiveness variable is 0.0 for orthographic, 1.0 for perspective, .5 for intermediate
    mesh_axis = Axis3(mesh_fig[1,1], xtickcolor=white,
                      viewmode=:fit, aspect=(1,1,1), perspectiveness=0.0, protrusions=0, limits=lim)
    if mesh_or_wire == "wire"
        wireframe!(mesh_axis, shape, color=:black)
    elseif mesh_or_wire == "mesh"
        mesh!(mesh_axis, shape, color=:skyblue2)
    end
    meshscene = mesh_axis.scene[end]
    GLMakie.rotate!(meshscene, rotation)
    hidedecorations!(mesh_axis)
    hidespines!(mesh_axis)
    cam = cam3d!(mesh_axis.scene)
    cam.projectiontype[] = AbstractPlotting.Orthographic
    cam.upvector[] = Vec3f0(1, 0, 0)
    cam.lookat[] = Vec3f0(0, 0, 0)
    cam.eyeposition[] = Vec3f0(0, 700, 0)
    update_cam!(mesh_axis.scene, cam)
    return mesh_fig
end    

function scene_to_matrix(mesh_fig)
    gray_grid = Gray.(GLMakie.scene2image(mesh_fig.scene)[1].parent)
    #    gray_grid = Gray.(AbstractPlotting.colorbuffer(mesh_fig.scene))
    # these run at the same speed but have different rotations in the end.
    # also, colorbuffer shows the image. 
    gray_matrix = convert(Matrix{Float64}, gray_grid')
    return gray_matrix
#    return gray_matrix'
end

function animate_mesh_rotation(shape, rotations)
    time_node = Node(1);
    f(t, rotations) = qrotation(rotations[t]...)
    mesh_fig, mesh_axis = GLMakie.wireframe(shape, color=:black)
    meshscene = mesh_axis.scene[end]
    # another option is rotating outside the "rotations" call and instead lifting the mesh itself.
    # then all of your rotations are on the mesh instead of the 
    screen = display(mesh_fig)
    # REPLACE THIS WITH AXIS3
    remove_axis_from_scene(mesh_axis)
    for r in rotations
        GLMakie.rotate!(meshscene, qrotation(r...))
        # need a 2D gridsave in the loop
        sleep(.1)
    end
    return mesh_axis
end


function remove_axis_from_scene(mesh_axis)
    white = RGBAf0(255, 255, 255, 0.0)
    threeDaxis = mesh_axis.scene
#    threeDaxis = mesh_axis.scene[OldAxis]
    threeDaxis[:showgrid] = (false, false, false)
    threeDaxis[:showaxis] = (false, false, false)
    threeDaxis[:ticks][:textcolor] = (white, white, white)
    threeDaxis[:names, :axisnames] = ("", "", "")
    return threeDaxis
end


function makie_plot_grid(g::UniformPointPushforwardGrid,
                         x_addr, y_addr;
                         title::String="Cell weights and representative points")
    @assert x_addr != y_addr
    partitions = Dict(
        x_addr => g.addr2partition[x_addr],
        y_addr => g.addr2partition[y_addr])
    repss = Dict()
    valss = Dict()
    sub_boundss = Dict()
    for (addr, prt) in partitions
        if prt isa DiscreteSingletons
            valss[addr] = all_representatives(prt)
            repss[addr] = 1:length(valss[addr])
            sub_boundss[addr] = 1//2 : 1 : length(valss[addr]) + 1//2
        else
            repss[addr] = all_representatives(prt)
            sub_bounds = all_subinterval_bounds(prt)
            clip_to_finite!(sub_bounds, lower=minimum(repss[addr]) - 5,
                            upper=maximum(repss[addr]) + 5)
            sub_boundss[addr] = sub_bounds
        end
    end
    w = let w_ = GenGridEnumeration.weights(g)
        addrs = collect(keys(g.addr2partition))
        (x_ind, y_ind) = indexin([x_addr, y_addr], addrs)
        dims = Tuple(setdiff(1:length(addrs), [x_ind, y_ind]))
        w = dropdims(sum(w_; dims=dims); dims=dims)
        x_ind < y_ind ? w : w'
    end
    (x_heavy, y_heavy) = let (i_x, i_y) = Tuple(argmax(w))
        (repss[x_addr][i_x], repss[y_addr][i_y])
    end
    println("making heatmap")
    f = Figure(resolution = (1600, 800))
    ax = GLMakie.Axis(f[1, 1])
    ax.xlabel = string(x_addr)
    ax.ylabel = string(y_addr)
    if x_addr ∈ keys(valss)
        ax.xticks = (1:length(valss[x_addr]), [string(round(v, digits=2)) for v in valss[x_addr]])
    end
    if y_addr ∈ keys(valss)
        ax.yticks = (1:length(valss[y_addr]), [string(round(v, digits=2)) for v in valss[y_addr]])
    end
    heatmap!(ax, float(collect(sub_boundss[x_addr])), float(collect(sub_boundss[y_addr])), w', colormap=:thermal)
    display(f)
    return ax
end


    
struct NoisyMatrix <: Gen.Distribution{Matrix{Float64}} end

const noisy_matrix = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Matrix{Float64}, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function Gen.random(::NoisyMatrix, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    mat = copy(mu)
    (w, h) = size(mu)
    for i=1:w
        for j=1:h
            mat[i, j] = mu[i, j] + randn() * noise
            if mat[i, j] > 1
                mat[i, j] = 1
            elseif mat[i, j] < 0
                mat[i, j] = 0
            end
        end
    end
    return mat
end


""" NOTES """
# cube_mesh.position yields the unrotated vertices
# cube.transformation.rotation gives you the rotation applied.
# cube.transformation.scale gives you scaling
# qrotation takes an axis (e.g. Vec3(1,0,0)) and a radian angle and returns a Quaternion. 
# w Rotations.jl, can make a Quat(q.data...) call to get a quaternion, which can be multiplied by 3D vecs
# coordinates(shape, nvertices=2) returns 
# ax.scene.camera contains the projection matricies w/ .projeciton and .projectionview
# lift syntax for animating a meshscatter

    # mesh_fig, mesh_ax = meshscatter(
    #     Point3f0.(rand.() .* 4, rand(N) .* 4, rand.() .* 0.01),
    #     markersize = Vec3f0.([0, .3], [0, .3], [0, 1.0]), 
    #     marker = cube_prim, 
    #     color = :skyblue2,
    #     rotations =  lift(t -> f(t, rotations), time_node),
    #     ssao = true)

# ax.scene.center = false
# im = GLMakie.scene2image(ax.scene)
#     # EQUIVALENT
# save("test.png", ax.scene)
# save("test2.png", im[1])


# FUNCTIONS FOR ROTATING POINTS OF A MESH
# z_rotator = qrotation(Vec3f0(0, 0, 1), -.5)
# x_rotator = qrotation(Vec3f0(1, 0, 0), -.5)
# rotator = z_rotator * x_rotator
# rotated_verts = [rotator * v for v in vertices]
# rotated_mesh = GeometryBasics.Mesh(rotated_verts, faces)

# FUNCTIONS FOR DECOMPOSING GB SHAPES INTO VERTICES AND FACES
#vertices = decompose(Point{3, Float64}, cube_prim)
# list of vertices. can also make a mesh and .position
#faces = decompose(TriangleFace{Int}, cube_prim)
# this is a list that connects indices of vertices to each other w/ a triangle

#cube_mesh = GeometryBasics.Mesh(vertices, faces)

# for speed i think we want to switch to colorbuffer(scene) instead of screen2image.
# for screen to image, you have to display the scene, which takes time. 




