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

cube_prim = Rect(Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))
cube_mesh = GeometryBasics.mesh(cube_prim)
vertices = cube_mesh.position
rotator = Quaternionf0(0, -.2, 0, 1)

# RENDER THE WIREFRAMES AND MESH PLOTS DIRECTLY FROM CUBE_PRIM.
# TRANSFORM POSITIONS USING [rotator * p for p in cube_mesh.position]
# use 


fig, ax = mesh(
    cube_mesh, color = :skyblue2,
)



 
cube = ax.scene[end]
GLMakie.rotate!(cube, rotator)

rotated_positions = [rotator * p for p in cube_mesh.position]

# once you've appled the same rotations to the mesh in the plot and the


# there is a 3 column matrix that comes out of this that changes w rotation. 
println(cube.model)
# cube_mesh.position yields the unrotated vertices
# cube.transformation.rotation gives you the rotation applied.
# cube.transformation.scale gives you scaling
# 




#ax.scene[end] is the rectangle. you can use this in the rotation function rotate! and the axis will stay the same
# but the cube will rotate. 


fig, ax = mesh(
 Rect(Vec3f0(-0.5), Vec3f0(1)), color = :skyblue2,
)

# ax.scene.camera contains the projection matricies w/ .projeciton and .projectionview

screen = display(ax.scene)
two_d_grid = AbstractPlotting.colorbuffer(screen)

# THIS REQUIRES DISPLAY |^

# This doesnt

# GLMakie.scene2image(ax.scene)

# apparently you can also disable the rendering loop to make colorbuffer faster
# GLMakie.opengl_renderloop[] = (screen) -> nothing. would have to test this. 





# to get a 2D grid out of the campixel


#rect = scene[end] # last plot is the rect
# there are a couple of ratate! functions, that accept e.g. a vector etc
#rotate!(rect, Quaternionf0(0, 0.4, 0, 1))
#scene

