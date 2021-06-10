using GLMakie
using Gen
using LinearAlgebra
using ImageFiltering


# eventually want to find a way to subtype the voxel types to define
# voxel-wide fields,
# repeat definition of fields and to define functions on Voxels (i.e. adding, preventing
# collision)

# ALSO note the illusion probably comes from the floor perspective.
# play with the camera location, ideas about perspective. things that are further will be
# further from the ground.


spherical_overlap(vox::SphericalVoxel, coord::Tuple) = coord[1] == vox.az && coord[2] == vox.alt && coord[3] == vox.r

CoordDivs = Dict(:az => collect(-50:10:50), 
                 :alt => collect(-50:10:50),
                 :dist => collect(0:5))

photon_count_probs(n, max) = .5 ^(n) * [i <= n ? binomial(n, i) : 0. for i=0:max]
onehot(v, dom) = [x == v ? 1 : 0 for x in dom]
@dist LabeledCat(labels, pvec) = labels[categorical(pvec)]
findnearest(input_arr::AbstractArray, val) = input_arr[findmin([abs(a) for a in input_arr.-val])[2]]

# start the recursion with dist_i = length(distance_divs)


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

# OK so now you have to make a model that generates objects in XYZ space and makes an
# azimuth altitude representation of them. so you generate an XYZ object,
# pass that to the spherical transform, then project that onto the retina.
# your latent states will contain a distance, az and alt, but the likelihood is performed
# on the az alt overlap. then write a program like move_step from energy model that moves the objects

# maybe you actually want to just start with a particular xyz coord for now.

# note you may actually want to infer photon counts behind occluders sometimes; depends on if you want to
# infer "brightnesses" or "objects"


@gen function generate_image()
    # eventually draw more complex shapes
    shape = LabeledCat([:rect], [1])
    shape_height = { :shape_height } ~  uniform_discrete(1, 5)
    brightness = { :brightness } ~ uniform_discrete(50, 100)
    alpha = { :alpha } ~ uniform_discrete(1, 1)
    shape_x = { :shape_x } ~ uniform_discrete(1, 10)
    shape_y = { :shape_y } ~ uniform_discrete(1, 10)
    shape_z = { :shape_z } ~ uniform_discrete(1, 10)
    object_vox = []
    if shape == :line
        for z in shape_z:shape_height
            vox = XYZVoxel(shape_x, shape_y, z, alpha, brightness)
            push!(object_vox, vox)
        end
    end
    # shape dist is implicitly the mag of the vector between the eye and the vox
    object_xyz = Object3D(object_vox)
    # deterministic for now but want to infer cam angle too
    eye = Detector(0, 0, 0, [1, 0, 0], [0, 1, 0], [0, 0, 1])
    object_in_spherical = xyz_vox_to_spherical([object_xyz], eye)
    az_alt_retina = { :image_2D } ~ produce_noisy_2D_retinal_image(set_world_state(object_in_spherical))
    return az_alt_retina
end


function recur_lightproj_deterministic(az_i, alt_i, dist_i, world_state, radial_photon_count)
    if dist_i == 0
        return convert(Float64, radial_photon_count)
    else
        tile_occupied = map(ws -> spherical_overlap(
            ws,
            (CoordDivs[:az][az_i], CoordDivs[:alt][alt_i], CoordDivs[:dist][dist_i])),
                            world_state)
        if any(tile_occupied)
            vox = world_state[findfirst(tile_occupied)]
            # note all filters in the world are multiplicative and not additive.
            photon_count_tile = vox.brightness
            photon_count_multiplier = 1-vox.alpha
        else
            photon_count_tile = 0
            photon_count_multiplier = 1
        end
        recur_lightproj_deterministic(az_i, alt_i, dist_i - 1,
                                      world_state,
                                      (radial_photon_count * photon_count_multiplier) + photon_count_tile)
    end
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
                         

@gen function produce_noisy_2D_retinal_image(world_state::Array{Voxel})
    projected_az_alt_vals = [recur_lightproj_deterministic(az_i, alt_i, length(CoordDivs[:dist]), world_state, 0) for az_i=1:length(CoordDivs[:az]), alt_i=1:length(CoordDivs[:alt])]

    az_alt_mat = reshape(projected_az_alt_vals, (length(CoordDivs[:az]), length(CoordDivs[:alt])))
    noisy_projection ~ generate_bitnoise(az_alt_mat, 3)
end
                                                                                       

function make_2D_constraints(input_trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    constraints = Gen.choicemap()
    set_submap!(constraints, :image_2D, get_submap(get_choices(input_trace), :image_2D))
    tr, w = Gen.generate(generate_image, (), constraints)
    return tr
end


function set_world_state(objects)
    world_state = []
    for object in objects
        world_state = vcat(world_state, object.voxels...)
    end
    return convert(Array{Voxel}, world_state)
end

# method for this is in master.py in PreyCapMaster line 2000.
            # spherical bout to XYZ is how i moved the fish in the virtual world.
            

# probably want to make this stochastic, reflecting any noise in the transfer from model to tectum. 
            
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
    tr = Gen.simulate(produce_2D_retinal_image, (world_state_spherical, ))
    return tr
end


function retina_mh_update(tr_populated, amnt_computation)
    mh_traces = []
    accepted_list = []
    tr = make_2D_constraints(tr_populated)
    for i in 1:amnt_computation
        (tr, accepted) = Gen.mh(tr, select(:shape_x, :shape_y, :shape_z, :shape_height))
        push!(mh_traces, tr)
        push!(accepted_list, accepted)
    end
    max_distance = floor(maximum([norm([x,y,z]) for x in 1:10, y in 1:10, z in 1:10]))
    dist_vs_height = zeros(length(1:max_distance), 5)
    for mht in mh_traces
        dist = convert(Int64, floor(norm([tr[:shape_x], tr[:shape_y], tr[:shape_z]])))
        height = tr[:shape_height]
        dist_vs_height[dist, height] += 1
    end
    fig = Figure()
    ax_dist_height = fig[1, 1] = Axis(fig)
    heatmap!(ax_dist_height, dist_vs_height)
    display(fig)
    return mh_traces, accepted_list
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





# use this if i you want to model the image generation process at the photon level. 

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


@gen function produce_2D_retinal_image(world_state::Array{Voxel})
    projected_vals = [{(az_i, alt_i)} ~ recur_light_projection(az_i, alt_i, length(CoordDivs[:dist]), world_state, 0)
                      for az_i=1:length(CoordDivs[:az]),
                          alt_i=1:length(CoordDivs[:alt])]
    return projected_vals
end

    
    


