using GLMakie
using Gen
using GenGridEnumeration
using OrderedCollections
#using LinearAlgebra
using Random
using Statistics
using StatsBase
using GeometryBasics
using Images

# note there are residual types in Makie from GeometryTypes, which is
# deprecated in favor of GeometryBasics

# Want to eventually do a type definition for shapes. will have vertices and faces as fields,
# will redefine GLMakie.mesh and wireframe to take Shape objects and render them.
# will be able to compose composite shapes using the kernel func type trees. also then
# assign translations and rotations using the function composition from pointlight. 



@dist labeled_cat(labels, weights) = labels[categorical(weights)]

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

    
function enumeration_inference(two_d_projection)
    constraints = Gen.choicemap((:image_2D, two_d_projection))
    tr, w = Gen.generate(primitive_shapes, (), constraints)
    g = UniformPointPushforwardGrid(tr, OrderedDict(
        :shape_choice => DiscreteSingletons([:cube, :tetrahedron]),
        :side_length => DiscreteSingletons([1, 2]),
        :rot_x => DiscreteSingletons(collect(0:.1:π)),
        :rot_y => DiscreteSingletons(collect(0:.1:π)),
        :rot_z => DiscreteSingletons(collect(0:.1:π))))
    return g
end
    
    

    
    



# can also get positions w/out going to mesh first if you want to. i.e. vertices = decompose(Point3, cube_prim)
# can construct a primitive as well w/ a set of vertices (i.e. 

#camera(mesh_axis.scene).eyeposition

sphere_prim = GeometryBasics.Sphere(Point3(0.0, 0.0, 0.0), 1) # origin, radius
cylinder_prim = Cylinder(Point3(0.0, 0.0, 0.0), Point3(0.0,0.0,1.0), 1.0) #(origin, normal vector, width)

function shape_wrap()
    trace = Gen.simulate(primitive_shapes, ())
    fig = Figure()
    mesh_axis, projected_grid, shape = get_retval(trace)
    #    fig[1,1] = mesh_axis
    
    #    two_d_grid = fig[1,2]
    # have to add image! to the figure
#    im = Gray.(projected_grid)
    save("test.png", projected_grid)
    display(mesh_axis.scene)
    return shape, mesh_axis
end
                                   
@gen function primitive_shapes()
    shape_type = { :shape_choice } ~ labeled_cat([:cube, :tetrahedron], [1/2, 1/2])
    side_length = { :side_length } ~ uniform_discrete(1,2)
    if shape_type == :cube
        shape = make_cube_mesh(convert(Float64, side_length))
    elseif shape_type == :tetrahedron
        shape = make_tetrahedron_mesh(convert(Float64, side_length))
    end
    rotation_x = { :rot_x } ~ uniform(0, 0)
    rotation_y = { :rot_y } ~ uniform(0, 0)
    rotation_z = { :rot_z } ~ uniform(0, π)
    axis3_vectors = [Vec(1.0, 0.0, 0.0),
                     Vec(0.0, 1.0, 0.0),
                     Vec(0.0, 0.0, 1.0)]
    quat_rotations = [qrotation(v, r) for (v, r) in zip(
        axis3_vectors, [rotation_x, rotation_y, rotation_z])]
    rotation_quaternion = reduce(*, quat_rotations)
    mesh_ax = render_static_mesh(shape, rotation_quaternion, "wire")
    mesh_ax.scene.center = false
#    display(mesh_ax.scene)
 #   sleep(5)
    projected_grid = scene_to_matrix(mesh_ax)
    noisy_image = ({ :image_2D } ~ noisy_matrix(projected_grid, 0.1))
    return mesh_ax, noisy_image, shape
end

function render_static_mesh(shape, rotation::Quaternion{Float64}, mesh_or_wire::String)
    white = RGBAf0(255, 255, 255, 0.0)
    if mesh_or_wire == "wire"
        mesh_fig, mesh_axis = GLMakie.wireframe(shape, color=:black)
    elseif mesh_or_wire == "mesh"
        mesh_fig, mesh_axis = GLMakie.mesh(shape, color=:skyblue2)
    end
    meshscene = mesh_axis.scene[end]
    screen = display(mesh_fig)
    remove_axis_from_scene(mesh_axis)
    rotate!(meshscene, rotation)
    return mesh_axis
end    

function scene_to_matrix(mesh_ax)
    gray_grid = Gray.(GLMakie.scene2image(mesh_ax.scene)[1].parent)
    gray_matrix = zeros(size(gray_grid)[2], size(gray_grid)[1])
    for i in 1:size(gray_grid)[1]
        for j in 1:size(gray_grid)[2]
            gray_matrix[j, i] = convert(Float64, gray_grid[i, j])
        end
    end
    return gray_matrix
end

function animate_mesh_rotation(shape, rotations)
    time_node = Node(1);
  
    f(t, rotations) = qrotation(rotations[t]...)
    mesh_fig, mesh_axis = GLMakie.wireframe(shape, color=:black)
    meshscene = mesh_axis.scene[end]
    # another option is rotating outside the "rotations" call and instead lifting the mesh itself.
    # then all of your rotations are on the mesh instead of the 
    screen = display(mesh_fig)
    remove_axis_from_scene(mesh_axis)
    for r in rotations
        rotate!(meshscene, qrotation(r...))
        # need a 2D gridsave in the loop
        sleep(.1)
    end
    return mesh_axis
end


function remove_axis_from_scene(mesh_axis)
    white = RGBAf0(255, 255, 255, 0.0)
    threeDaxis = mesh_axis.scene[OldAxis]
    threeDaxis[:showgrid] = (false, false, false)
    threeDaxis[:showaxis] = (false, false, false)
    threeDaxis[:ticks][:textcolor] = (white, white, white)
    threeDaxis[:names, :axisnames] = ("", "", "")
    return threeDaxis
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
