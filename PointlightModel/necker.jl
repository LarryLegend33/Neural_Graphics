using GLMakie
using Gen
using GenGridEnumeration
#using LinearAlgebra
using Random
using Statistics
using StatsBase
using GeometryBasics

# note there are residual types in Makie from GeometryTypes, which is
# deprecated in favor of GeometryBasics

N = 100

# m = meshscatter(
#     Point3f0.(rand.() .* 4, rand(N) .* 4, rand.() .* 0.01),
#     markersize = Vec3f0.(rand.() .* 0.3, rand.() .* 0.3, rand(N)), 
#     marker = FRect3D(Vec3f0(0), Vec3f0(1)),
#     color = rand(RGBf0, N),
#     rotations = Vec3f0.(rand(N) .* 0.3, rand.() .* 0.8, 1) # also accepts Quaternions
# )


# constructor has to be with floats to use quaternion rotation w/ floats. q * p

cube_prim = GeometryBasics.Rect(Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))

# can also get positions w/out going to mesh first if you want to. i.e. vertices = decompose(Point3, cube_prim)
# can construct a primitive as well w/ a set of vertices (i.e. 


#cube_mesh = GeometryBasics.mesh(cube_prim)
#cube_mesh.position

vertices = decompose(Point{3, Float64}, cube_prim)
# list of vertices
faces = decompose(TriangleFace{Int}, cube_prim)
# this is a list that connects indices of vertices to each other w/ a triangle

cube_mesh = GeometryBasics.Mesh(vertices, faces)

z_rotator = qrotation(Vec3f0(0, 0, 1), -.5)
x_rotator = qrotation(Vec3f0(1, 0, 0), -.5)
rotator = z_rotator * x_rotator


#rotator = Quaternionf0(0, -.2, 0, 1)

rotated_verts = [rotator * v for v in vertices]
rotated_mesh = GeometryBasics.Mesh(rotated_verts, faces)

fig, ax = GLMakie.mesh(rotated_mesh, color=:skyblue2)

# qrotate will make a quaternion out of a set of rotations
# 

# RENDER THE WIREFRAMES AND MESH PLOTS DIRECTLY FROM CUBE_PRIM.
# TRANSFORM POSITIONS USING [rotator * p for p in cube_mesh.position]
# use 

    
# w Rotations.jl, can make a Quat(q.data...) call to get a quat
    
# fig, ax = mesh(
#     cube_prim, color = :skyblue2,
# )
 
cube = ax.scene[end]
GLMakie.rotate!(cube, rotator)

#rotated_positions = [rotator * p for p in cube_mesh.position]

# once you've appled the same rotations to the mesh in the plot and the


# there is a 3 column matrix that comes out of this that changes w rotation. 
println(cube.model)
# cube_mesh.position yields the unrotated vertices
# cube.transformation.rotation gives you the rotation applied.
# cube.transformation.scale gives you scaling
# 

# qrotation takes an axis (e.g. Vec3(1,0,0)) and a radian angle and returns a Quaternion. 


#ax.scene[end] is the rectangle. you can use this in the rotation function rotate! and the axis will stay the same
# but the cube will rotate. 




# ax.scene.camera contains the projection matricies w/ .projeciton and .projectionview

screen = display(ax.scene)
two_d_grid = AbstractPlotting.colorbuffer(screen)

    

# THIS REQUIRES DISPLAY |^

# This doesnt



#can also


# This completely works    
ax.scene.center = false
im = GLMakie.scene2image(ax.scene)
    # EQUIVALENT
save("test.png", ax.scene)
save("test2.png", im[1])



# apparently you can also disable the rendering loop to make colorbuffer faster
# GLMakie.opengl_renderloop[] = (screen) -> nothing. would have to test this. 





# to get a 2D grid out of the campixel


#rect = scene[end] # last plot is the rect
# there are a couple of ratate! functions, that accept e.g. a vector etc
#rotate!(rect, Quaternionf0(0, 0.4, 0, 1))
#scene

