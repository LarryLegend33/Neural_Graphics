using GLMakie
using Gen
using LinearAlgebra


# eventually want to find a way to subtype the voxel types to define
# voxel-wide fields,
# repeat definition of fields and to define functions on Voxels (i.e. adding, preventing
# collision)

abstract type Voxel end

struct SphericalVoxel <: Voxel
    az::Float64
    alt::Float64
    r::Int64
    alpha::Float64
    brightness::Float64
    # color::
end

struct XYZVoxel <: Voxel
    x::Float64
    y::Float64
    z::Int64
    alpha::Float64
    brightness::Float64
end

struct Detector
    x::Float64
    y::Float64
    z::Float64
    u_lens::Array{Float64}
    u_horiz::Array{Float64}
    u_normal::Array{Float64}
    #pitch::Float64
    #yaw::Float64
end


# eventual goal is to write methods to move Object3Ds through
# space

struct Object3D
    voxels::Array{Voxel}
    # eventually want to give each voxel material a color
end


spherical_overlap(vox::SphericalVoxel, coord::Tuple) = coord[1] == vox.az && coord[2] == vox.alt && coord[3] == vox.r

const CoordDivs = Dict(:az => collect(-50:5:50), 
                       :alt => collect(-50:5:50),
                       :dist => collect(0:20))

photon_count_probs(n, max) = .5 ^(n) * [i <= n ? binomial(n, i) : 0. for i=0:max]
onehot(v, dom) = [x == v ? 1 : 0 for x in dom]
@dist LabeledCat(labels, pvec) = labels[categorical(pvec)]
findnearest(input_arr::AbstractArray, val) = input_arr[findmin([abs(a) for a in input_arr.-val])[2]]

# start the recursion with dist_i = length(distance_divs)

@gen function recur_light_projection(az_i, alt_i, dist_i, world_state, radial_photon_count)
    noise_baseline = .01
    if dist_i == 0
        return radial_photon_count
    else
        tile_occupied = map(ws -> spherical_overlap(
            ws,
            (CoordDivs[:az][az_i], CoordDivs[:alt][alt_i], CoordDivs[:dist][dist_i])),
                            world_state)
        if any(tile_occupied)
            vox = world_state[findfirst(tile_occupied)]
            # note all filters in the world are multiplicative and not additive.
            photon_count_tile = { (:photons, dist_i) } ~ Gen.normal(vox.brightness, .1)
            photon_count_multiplier = { (:photon_mult, dist_i) } ~ Gen.beta(1-vox.alpha+noise_baseline,
                                                                            vox.alpha+noise_baseline)
        else
            photon_count_tile = { (:photons, dist_i) } ~ Gen.normal(0, .1)
            photon_count_multiplier = { (:photon_mult, dist_i) } ~ Gen.beta(1, noise_baseline)
        end
        {*} ~ recur_light_projection(az_i, alt_i, dist_i - 1,
                                     world_state,
                                     (radial_photon_count * photon_count_multiplier) + photon_count_tile)
    end
end
    

@gen function produce_retinal_image(world_state::Array{Voxel})
    projected_vals = [{(az_i, alt_i)} ~ recur_light_projection(az_i, alt_i, length(CoordDivs[:dist]), world_state, 0)
                      for az_i=1:length(CoordDivs[:az]),
                          alt_i=1:length(CoordDivs[:alt])]
    return projected_vals
end


    # world state should be a dictionary of transparencies at each az alt location.
    # inputs should be a list of 3DObjects
    # dont make a full grid -- unnecessary. live in a world of objects. 
    
function set_world_state(objects)
    world_state = []
    for object in objects
        world_state = vcat(world_state, object.voxels...)
    end
    return convert(Array{Voxel}, world_state)
end


# OK the last step is to make a 2D projection out of this. currently getting the correct az, alt results.
# use a perspective projection. if you do, the XY rendering of something moving backwards will move sideways.


# could consider having world state be an xyz position of each object
# then translate those to a SpericalVoxel state based on the position of the
# camera. 


# method for this is in master.py in PreyCapMaster line 2000.
# spherical bout to XYZ is how i moved the fish in the virtual world. 

function xyz_vox_to_spherical(objects_in_xyz::Array{Object3D}, detector::Detector)

    # output of this function is a list of objects containing sphericalvoxels
    objects_in_spherical = []
    for object in objects_in_xyz
        xyz_voxels = object.voxels
        spherical_voxels = []
        for vox in xyz_voxels
            pos_wrt_detector = [vox.x - detector.x, vox.y - detector.y, vox.z - detector.z]
            basis_wrt_detector_lens = [dot(pos_wrt_detector, detector.u_lens),
                                       dot(pos_wrt_detector, detector.u_horiz),
                                       dot(pos_wrt_detector, detector.u_normal)]
            radius = norm(basis_wrt_detector_lens)
            vox_vector_unit = basis_wrt_detector_lens / radius
            alt = asin(vox_vector_unit[3])
            az = atan(vox_vector_unit[2] / vox_vector_unit[1])
            if vox_vector_unit[1] < 0
                if az < 0
                    az = π + az
                else
                    az = -π + az
                end
            end
            # eventually want an "out of range" filter here for if any of the az, alt, dist coord are
            # not in the div range, meaning you can't see the object. 
            push!(spherical_voxels, SphericalVoxel(findnearest(CoordDivs[:az], rad2deg(az)),
                                                   findnearest(CoordDivs[:alt], rad2deg(alt)),
                                                   findnearest(CoordDivs[:dist], radius),
                                                   vox.alpha, vox.brightness))
        end
        push!(objects_in_spherical, Object3D(spherical_voxels))
    end
    return objects_in_spherical
end



function retina_proj_wrap()
    xyz_object_state = [Object3D([XYZVoxel(2, 20, 2, 1, 100), XYZVoxel(2, 20, 0, 1, 100)])]
    eye = Detector(0, 0, 0, [1, 0, 0], [0, 1, 0], [0, 0, 1])
    spherical_object_state = xyz_vox_to_spherical(xyz_object_state, eye)
    world_state_spherical = set_world_state(spherical_object_state)
    tr = Gen.simulate(produce_retinal_image, (world_state_spherical, ))
    return tr
end

# OK! seems like this all works in principle. figure out what you want your inference results to feel like tomorrow



        
# only need this to go back to an XYZ world location. this is for SLAM or something like that; i.e. an allocentric position
# in xyz world space
        
#function project_to_xyz(trace::, cam::Detector)
    
    # z_in_basis = r * sin(alt)
    # y_in_basis = r * cos(alt) * cos(az)
    # x_in_basis = r * cos(alt * sin(az)
    # draw these out perfectly later.
    # unit_cam, unit_parallel, unit_perp has to be calculated from cam_unit_vec
    # dx = unit_cam * x_in_basis
    # dy = unit_par * y_in_basis
    # dz = unit_perp * z_in_basis
    # final xyz coord = cam.x + dx, cam.y + dy, cam.z + dz
    
    # 
#end




    
    
    
    


