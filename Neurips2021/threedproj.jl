using GLMakie
using Gen
using LinearAlgebra
using ImageFiltering
using OrderedCollections
using Distributions
using StatsBase
using Colors


#i might get rid of 0 velocity

#and decrease the amount of steps significantly

#(to maybe like 5 steps)

# eventually want to find a way to subtype the voxel types to define
# voxel-wide fields,
# repeat definition of fields and to define functions on Voxels (i.e. adding, preventing
# collision)

# ALSO note the illusion probably comes from the floor perspective.
# play with the camera location, ideas about perspective. things that are further will be
# further from the ground.

# start the recursion with dist_i = length(distance_divs)


# TODO

# in line_proposal write the correct way to get an exact az and alt coord out of a tile (median? mean?)



abstract type Voxel end

struct SphericalVoxel <: Voxel
    az::Float64
    alt::Float64
    r::Float64
    alpha::Float64
    brightness::Float64
    # color::
end

struct XYZVoxel <: Voxel
    x::Float64
    y::Float64
    z::Float64
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


# note that proposals will go from az, alt only to XYZ. estimates will be noisier for more distant grid tiles.

# first proposal is this is the position based on pixels.
# can propose what it is / how many objects there idea.

# threshold. next, calculate maybe_one_of(az). altitude is going to be spread over
# multiple divs; once you propose a distance, then this becomes a basically
# deterministic inversion to XYZ. we did a subtraction on the inferred x coordinate. 

# note for SMC, you'll need to propose the first distance. 


            

struct Object3D
    voxels::Array{Voxel}
    # eventually want to give each voxel material a color
end

#spherical_overlap(vox::SphericalVoxel, coord::Tuple) = coord[1] == vox.az && coord[2] == vox.alt && coord[3] == vox.r

spherical_overlap(vox::SphericalVoxel,
                  tile) = within(vox.az, tile[1], SphericalTiles[:az][end][end]) && within(
                      vox.alt, tile[2], SphericalTiles[:alt][end][end]) && within(
                          vox.r, tile[3], SphericalTiles[:r][end][end])

CoordDivs = Dict(:az => collect(-80.0:10.0:80.0), 
                 :alt => collect(-80.0:10.0:80.0),
                 :x => collect(2.0:20.0), 
                 :y => collect(-10.0:10.0),
                 :z => collect(-10.0:2.0:10.0),
                 :v => collect(2.0:4.0),
                 :height => collect(2.0:20.0),
                 :brightness => collect(100.0:100.0))

CoordDivs[:r] = collect(0:ceil(maximum([norm([x,y,z]) for x in CoordDivs[:x],
                                            y in CoordDivs[:y],
                                            z in CoordDivs[:z]])))
                           
SphericalTiles = Dict(:az => mapwindow(collect, CoordDivs[:az], 0:1, border=Inner()),
                      :alt => mapwindow(collect, CoordDivs[:alt], 0:1, border=Inner()),
                      :r => mapwindow(collect, CoordDivs[:r], 0:1, border=Inner()))

XInit = 10.0
YInit = -8.0
ZInit = 0.0
VThatLeadToXYInit = 1.0
HeightInit = 12.0

# only value that isn't caught by boundaries is the last member of the range.
# if the value is the last member of the domain and the last tile is the boundary. 
within(val, boundaries, b_max) = (val >= boundaries[1] && val < boundaries[end]) || (val == b_max && val == boundaries[end])
@dist LabeledCat(labels, pvec) = labels[categorical(pvec)]
@dist uniformLabCat(labels) = LabeledCat(labels, 1/length(labels) * ones(length(labels)))
findnearest(input_arr::AbstractArray, val) = input_arr[findmin([abs(a) for a in input_arr.-val])[2]]
even(input) = 2 * round(input / 2)

                            
onehot(x, dom) =
    x < first(dom) ? onehot(first(dom), dom) :
    x > last(dom)  ? onehot(last(dom), dom)  :
    [i == x ? 1. : 0. for i in dom]


# recoded these to account for divs between coords not being 1.
# want to use LabeledCat(CoordDivs[:az], maybe_one_off(az, .2, CoordDivs[:az])),
# with a findnearest(CoordDivs[:az], az) call. then put it in the right bin.
# could also code a custom maybe_one_off_bin which makes a onehot from a bin input
# and a maybe_one_off using the bin .- (dom[1][end]-dom[1][1]). actually this is good.
# Can also use LabeledCat(CoordDivs[:az], discretized_gaussian(az, std, CoordDivs[:az]))

maybe_one_bin_off(midval, prob, dom) =
    (1 - prob) * onehot(midval, dom) +
    prob/2 * onehot(midval .- (dom[1][end]-dom[1][1]), dom) +
    prob/2 * onehot(midval .+ (dom[1][end]-dom[1][1]), dom)

maybe_one_off(midval, prob, dom) =
    (1 - prob) * onehot(midval, dom) +
    prob/2 * onehot(midval - (dom[2]-dom[1]), dom) +
    prob/2 * onehot(midval + (dom[2]-dom[1]), dom)

maybe_one_or_two_off(midval, prob, dom) = 
    (1 - prob) * onehot(midval, dom) +
    prob/3 * onehot(midval - (dom[2]-dom[1]), dom) +
    prob/3 * onehot(midval + (dom[2]-dom[1]), dom) +
    prob/6 * onehot(midval - (2*dom[2]-dom[1]), dom) +
    prob/6 * onehot(midval + (2*dom[2]-dom[1]), dom)

normalize(v) = v / sum(v)
discretized_gaussian(mean, std, dom) = normalize([
    cdf(Normal(mean, std), i + .5) - cdf(Normal(mean, std), i - .5) for i in dom
        ])


# x is into page. y is horizontal. z is vertical. 

@gen function generate_static_image()
    # eventually draw more complex shapes
    shape = LabeledCat([:rect], [1])
    height = { :height } ~  uniformLabCat(CoordDivs[:height])
    brightness = { :brightness } ~ uniformLabCat(CoordDivs[:brightness])
    alpha = { :alpha } ~ uniform_discrete(1, 1)
    # this will eventually be the object's center, with the voxels defined by nishad's descriptions. 
    x = { :x } ~ uniformLabCat(CoordDivs[:x])
    y = { :y } ~ uniformLabCat(CoordDivs[:y])
    z = { :z } ~ uniformLabCat(CoordDivs[:z])
    origin_to_objectcenter_dist = Int64(floor(norm([x, y, z])))
    r = { :r } ~ LabeledCat(CoordDivs[:r], discretized_gaussian(origin_to_objectcenter_dist, 2.0, CoordDivs[:r]))
    object_vox = []
    if shape == :rect
        for (i, zc) in enumerate(z-(height/2):z+(height/2))
            vox = XYZVoxel(x, y, zc, alpha, brightness)
            # eventually want distance of each component like this. propose the distance of each vox. 
            # distance = { :r } ~ Gen.normal(norm([shape_x, shape_y, z]), .01)
            push!(object_vox, vox)
        end
    end
    # shape dist is implicitly the mag of the vector between the eye and the vox
    object_xyz = Object3D(object_vox)
    # deterministic for now but want to infer cam angle too
    eye = Detector(0, 0, 0, [1, 0, 0], [0, 1, 0], [0, 0, 1])
    object_in_spherical = xyz_vox_to_spherical([object_xyz], eye)
    az_alt_retina = { :obs } ~ produce_noisy_2D_retinal_image(set_world_state(object_in_spherical))
    return convert(Matrix{Float64},
                   reshape(az_alt_retina, (length(SphericalTiles[:az]), length(SphericalTiles[:alt]))))
end

# instead of az, alt, and dist indices, this will be az and alt and dist wins.

# switch spherical_overlap to asking if the coordinate is inside the passed tile.
# project_noisy_2D_image will cycle through indices just as before, but in tile_occupied here,
# index the tiles. 

function recur_lightproj_deterministic(az_i, alt_i, dist_i, world_state, radial_photon_count)
    if dist_i == 0
        return convert(Float64, radial_photon_count)
    else
        tile_occupied = map(ws -> spherical_overlap(
            ws,
            [SphericalTiles[:az][az_i], SphericalTiles[:alt][alt_i], SphericalTiles[:r][dist_i]]),
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



""" NOISE MODEL 1: BITNOISE ON A BOXFILTER """

@gen function bernoulli_noisegen(p::Float64)
    # i.e. if all 9 are black pixels, still have a .1 chance of turning it white
    baseline_noise = 0.0
    pix ~ bernoulli(p+baseline_noise)
    return pix
end

# note problem with this is that you will likely extend the percept upwards 

@gen function generate_bitnoise(im_mat::Matrix{Float64}, filter_size::Int)
    # if all 9 are white pixels, still have a .1 chance of going black.
    # this will be offset by the baseline noise the other way. 
    baseline_noise = .99
    conv_filter = hcat(zeros(filter_size, 1), ones(filter_size, 1), zeros(filter_size, 1) /
        (filter_size))
    #    image_2D = { :image_2D } ~ Gen.Map(bernoulli_noisegen)(imfilter(im_mat, conv_filter))
    image_2D = { :image_2D } ~ Gen.Map(bernoulli_noisegen)(im_mat / CoordDivs[:brightness][1])
end


""" NOISE MODEL 2: GAUSSIAN ON A BOXFILTER """

# discretized normal distribution here. 

@gen function gaussian_noisegen(μ::Float64, maxpix::Float64)
    σ = 10.0
    pixrange = collect(0.0:maxpix)
    pix ~ LabeledCat(pixrange, discretized_gaussian(μ, σ, pixrange))
    #    pix ~ normal(μ, σ)
    
end

@gen function generate_blur(im_mat::Matrix{Float64}, filter_size::Int)
    # if all 9 are white pixels, still have a .1 chance of going black.
    # this will be offset by the baseline noise the other way. 
    conv_filter = ones(filter_size, filter_size) / (filter_size^2)
    #    maxpix = maximum(im_mat)
    maxpix = reduce(+ , [CoordDivs[:brightness][end] for d in SphericalTiles[:r]])
    size_mat = size(im_mat)
    image_2D = { :image_2D } ~ Gen.Map(gaussian_noisegen)(
        imfilter(im_mat, conv_filter), maxpix*ones(size_mat[1]*size_mat[2]))
end
# it's firm here on no args to the Gen function. dont know how to get it there otherwise. 



@gen function produce_noisy_2D_retinal_image(world_state::Array{Voxel})
    projected_az_alt_vals = [recur_lightproj_deterministic(az_i, alt_i, length(SphericalTiles[:r]), world_state, 0) for az_i=1:length(SphericalTiles[:az]), alt_i=1:length(SphericalTiles[:alt])]
    az_alt_mat = reshape(projected_az_alt_vals, (length(SphericalTiles[:az]), length(SphericalTiles[:alt])))
#    noisy_projection ~ generate_blur(az_alt_mat, 1)
    noisy_projection ~ generate_bitnoise(az_alt_mat, 1)
end


""" SMC RELATED FUNCTIONS """

@gen function move_object_in_xyz(T::Int)
    x = XInit
    y = YInit
    v = VThatLeadToXYInit
    height = HeightInit
    z = ZInit
    # this should be implicit. your velocity before this was 0.
    # XInit and EInit are added to the plot as the first member of the array. 
    xs = []
    hs = []
    obss = []
    # draw z, draw height
    size_or_depth = { :size_or_depth } ~ uniformLabCat([:size, :depth])
#    z = { :z } ~ uniformLabCat(CoordDivs[:Z])
    brightness = { :brightness } ~ uniformLabCat(CoordDivs[:brightness])
    alpha = { :alpha } ~ uniform_discrete(1, 1)
    for t in 1:T
        (v, x, y, height, obs) = {t} ~ xyz_step_model(v, x, y, z, height, alpha, brightness, size_or_depth)
        push!(xs, x)
        push!(hs, height)
        push!(obss, obs)
    end
    return (xs, hs, obss,
            [convert(Matrix{Float64}, reshape(
                obs, (length(SphericalTiles[:az]), length(SphericalTiles[:alt])))) for obs in obss])
end

@gen function xyz_step_model(v_prev, x_curr, y_curr, z_curr, height_curr, alpha, brightness, size_or_depth)
    # make a step depending on your energy and x location, and your previous velocity
    v = { :v } ~ LabeledCat(CoordDivs[:v], maybe_one_off(v_prev, .2, CoordDivs[:v]))
    if size_or_depth == :depth
        height = { :height } ~ uniform_discrete(convert(Int64, height_curr), convert(Int64, height_curr))
        x = { :x } ~ LabeledCat(CoordDivs[:x], maybe_one_off(x_curr + v, .2, CoordDivs[:x]))
    else
        height = { :height } ~ LabeledCat(CoordDivs[:height], maybe_one_off(height_curr - v, .2, CoordDivs[:height]))
        x = { :x } ~ uniform_discrete(convert(Int64, x_curr), convert(Int64, x_curr))
    end
    y = { :y } ~ LabeledCat(CoordDivs[:y], maybe_one_off(y_curr + v, .2, CoordDivs[:y]))
    # this will eventually be the object's center, with the voxels defined by nishad's descriptions. 
    origin_to_objectcenter_dist = Int64(floor(norm([x, y, z_curr])))
    r = { :r } ~ LabeledCat(CoordDivs[:r],
                            discretized_gaussian(origin_to_objectcenter_dist, 2.0, CoordDivs[:r]))
    object_vox = []
    for (i, zc) in enumerate(z_curr-round(height/2):z_curr+round(height/2))
        vox = XYZVoxel(x, y, zc, alpha, brightness)
        # eventually want distance of each component like this. propose the distance of each vox. 
        push!(object_vox, vox)
    end
    # shape dist is implicitly the mag of the vector between the eye and the vox
    object_xyz = Object3D(object_vox)
    # deterministic for now but want to infer cam angle too
    eye = Detector(0, 0, 0, [1, 0, 0], [0, 1, 0], [0, 0, 1])
    object_in_spherical = xyz_vox_to_spherical([object_xyz], eye)
    projected_az_alt_vals = [recur_lightproj_deterministic(az_i, alt_i, length(SphericalTiles[:r]),
                                                           set_world_state(object_in_spherical), 0) for az_i=1:length(SphericalTiles[:az]),
                                                               alt_i=1:length(SphericalTiles[:alt])]
    az_alt_mat = reshape(projected_az_alt_vals, (length(SphericalTiles[:az]), length(SphericalTiles[:alt])))

    az_alt_retina = { :image_2D } ~ Gen.Map(bernoulli_noisegen)(az_alt_mat / CoordDivs[:brightness][1])
  #  az_alt_retina = { :image_2D } ~ produce_noisy_2D_retinal_image(set_world_state(object_in_spherical))
    return (v, x, y, height, az_alt_retina)

end


# should i propose size or depth here or is that cheating?

# Make sure these get translated to radians!!! That's why they're off. 

@gen function linefinder_proposal(curr_trace, obs, t)
    occupied_azalt = sort(findall(f -> f > 0, obs))
    az_location = mode([c[1] for c in occupied_azalt])
    az_tile = SphericalTiles[:az][az_location]
    alt_tile_low = SphericalTiles[:alt][minimum([c[2] for c in occupied_azalt if c[1] == az_location])]
    alt_tile_high = SphericalTiles[:alt][maximum([c[2] for c in occupied_azalt if c[1] == az_location])]
    alt_midpoint_rad = deg2rad(alt_tile_high[2] - alt_tile_low[1]) / 2
    az_rad = deg2rad(mean(az_tile))
    
    # can be a bit more intelligent about distance here if you want. should be somewhat related to amount of alt tiles 
    if t > 1
        r = { t => :r } ~ LabeledCat(CoordDivs[:r],
                                maybe_one_or_two_off(round(norm([curr_trace[t-1 => :x],
                                                           curr_trace[t-1 => :y], ZInit])), .1, CoordDivs[:r]))
    else
        r = { t => :r } ~ LabeledCat(CoordDivs[:r], maybe_one_off(round(norm([XInit, YInit, ZInit])), .2, CoordDivs[:r]))
    end
    x = { t => :x } ~ LabeledCat(CoordDivs[:x],
                            maybe_one_off(even(round(r * cos(alt_midpoint_rad) * sin(az_rad))),
                                          .2, CoordDivs[:x]))
    y = { t => :y } ~ LabeledCat(CoordDivs[:y],
                            maybe_one_off(even(round(r * cos(alt_midpoint_rad) * cos(az_rad))),
                                          .2, CoordDivs[:y]))
   # z = { t => :z } ~ LabeledCat(CoordDivs[:z], maybe_one_off(even(
    #    r * sin(alt_tile_low[2]-alt_tile_low[1])), .2, CoordDivs[:z]))
    if t > 1
        v = { t => :v } ~ LabeledCat(CoordDivs[:v], maybe_one_off(y-curr_trace[t-1 => :y], .2, CoordDivs[:v]))
    else
        v = { t => :v } ~ LabeledCat(CoordDivs[:v], maybe_one_off(y-YInit, .2, CoordDivs[:v]))
    end
    height = { t => :height } ~ LabeledCat(CoordDivs[:height],
                                      maybe_one_off(round(r * sin(deg2rad(mean(alt_tile_high) - mean(alt_tile_low)))), .2, CoordDivs[:height]))
    return x, y, height
end


function linepos_particle_filter(num_particles::Int, gt_trace::Trace, gen_function::DynamicDSLFunction{Any}, proposal)
    observations = get_retval(gt_trace)[4]
    obs_choices = [Gen.choicemap() for i in 1:length(observations)]
    [set_submap!(c, t => :image_2D, get_submap(get_choices(gt_trace), t => :image_2D)) for (t, c) in enumerate(obs_choices)]
    if proposal == ()
        state = Gen.initialize_particle_filter(gen_function, (1,), obs_choices[1], num_particles)
    else
        state = Gen.initialize_particle_filter(gen_function, (1,), obs_choices[1], proposal, (gt_trace, observations[1], 1), num_particles)
    end
    for t in 2:length(observations)
        obs = obs_choices[t]
        Gen.maybe_resample!(state, ess_threshold=num_particles)
        if proposal == ()
            Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs)
        else
            Gen.particle_filter_step!(state, (t,), (UnknownChange(),), obs, proposal, (observations[t], t))
        end
    end
  #  heatmap_pf_results(state, gt_trace)
    return state
end


function heatmap_pf_results(state, gt::Trace)
    gray_cmap = range(colorant"white", stop=colorant"gray32", length=6)
    true_depth = get_retval(gt)[1]
    true_height = get_retval(gt)[2]
    times = length(get_retval(gt)[1])
    depth_matrix = zeros(times, length(CoordDivs[:x]) + 1)
    height_matrix = zeros(times, length(CoordDivs[:height]) + 1)
    for t in 1:times
        for tr in state.traces
            depth_matrix[t, Int64(tr[t => :x])] += 1
            height_matrix[t, Int64(tr[t => :height])] += 1
        end
    end
    # also plot the true x values
    fig = Figure(resolution=(1200,1200))
    ax_depth = fig[1, 1] = Axis(fig)
    hm_depth = heatmap!(ax_depth, depth_matrix, colormap=gray_cmap)    
    cbar = fig[1, 2] = Colorbar(fig, hm_depth, label="N Particles")
    ax_height = fig[2, 1] = Axis(fig)
    hm_height = heatmap!(ax_height, height_matrix, colormap=gray_cmap)
    cbar2 = fig[2, 2] = Colorbar(fig, hm_depth, label="N Particles")
#    scatter!(ax, [o-.5 for o in observations], [t-.5 for t in 1:times], color=:skyblue2, marker=:rect, markersize=30.0)
    scatter!(ax_depth, [t-.5 for t in 1:times], [tx-.5 for tx in true_depth], color=:orange, markersize=20.0)
    scatter!(ax_height, [t-.5 for t in 1:times], [th-.5 for th in true_height], color=:orange, markersize=20.0)
    ax_depth.ylabel = "Depth"
    ax_height.ylabel = "Height"
    ax_height.xlabel = "Time"
    ylims!(ax_height, (0.0, CoordDivs[:height][end]))
    ylims!(ax_depth, (0.0, CoordDivs[:x][end]))
    println(countmap([tr[:size_or_depth] for tr in state.traces]))
#    vlines!(ax, HOME, color=:red)
    display(fig)
    return fig
end



# note to get realistic effect, height has to grow upwards and downwards, not from bottom z up. 

function animate_azalt_movement(tr)
    obs = get_retval(tr)[end]
    fig = Figure(resolution=(1000,1000))
    azalt_ax = fig[1,1] = Axis(fig)
    time = Node(1)
    hm(t) = obs[t]
    heatmap!(azalt_ax, CoordDivs[:az], CoordDivs[:alt], lift(t -> hm(t), time))
    azalt_ax.aspect = DataAspect()
    azalt_ax.xlabel = "Azimuth"
    azalt_ax.ylabel = "Altitude"
    display(fig)
    for i in 1:length(obs)
        time[] = i
        sleep(.2)
    end
end


function make_2D_constraints(input_trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    constraints = Gen.choicemap()
    set_submap!(constraints, :image_2D, get_submap(get_choices(input_trace), :image_2D))
    return constraints
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
            push!(spherical_voxels, SphericalVoxel(rad2deg(az),
                                                   rad2deg(alt),
                                                   radius,
                                                   vox.alpha, vox.brightness))
        end
        push!(objects_in_spherical, Object3D(spherical_voxels))
    end
    return objects_in_spherical
end

# output something you can sample from here.
# xyz coord then you have stochastic lookup table of az and alt and dist.
# this is the part that goes from xyz occupied space to az alt r occupied space.

# the complicated part is saying which spots in between vertices should be filled in.
# start with an xyz cube. then transform that to an overlap with az, alt, r grid.
# then select the exact tile of the az, alt r grid based on transforming overlaps to probabilities. 


function retina_proj_wrap()
    tr = Gen.simulate(generate_image, ())
    retina_proj_wrap(tr)
end

function retina_proj_wrap(constraints::Gen.DynamicDSLChoiceMap)
    tr, w = Gen.generate(generate_image, (), constraints)
    println(w)
    retina_proj_wrap(tr)
end

function retina_proj_wrap(tr::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    az_alt_grid = get_retval(tr)
    println("X")
    println(tr[:x])
    println("Y")
    println(tr[:y])
    println("Z")
    println(tr[:z])
    println("Height")
    println(tr[:height])
    fig = Figure(resolution=(1000,1000))
    azalt_ax = fig[1,1] = Axis(fig)
    heatmap!(azalt_ax, CoordDivs[:az], CoordDivs[:alt], az_alt_grid)
    azalt_ax.aspect = DataAspect()
    azalt_ax.xlabel = "Azimuth"
    azalt_ax.ylabel = "Altitude"
    display(fig)
    return tr
end


function retina_mh_update(tr_populated, amnt_computation)
    mh_traces = []
    accepted_list = []
    println(tr_populated[:height])
    cmap_w_image2D = make_2D_constraints(tr_populated)
    tr, w = Gen.generate(generate_image, (), cmap_w_image2D)
    for i in 1:amnt_computation
        (tr, accepted) = Gen.mh(tr, select(:x, :y, :z, :height))
        push!(mh_traces, tr)
        push!(accepted_list, accepted)
    end
    # 
    depth_vs_height = zeros(length(CoordDivs[:x]), length(CoordDivs[:height]))
    xvals = []
    for mht in mh_traces
        depth = mht[:x]
        height = tr[:height]
        depth_vs_height[findfirst(x -> x == depth, CoordDivs[:x]),
                       findfirst(h -> h == height, CoordDivs[:height])] += 1
        push!(xvals, tr[:x])
    end
    fig = Figure()
    ax_dist_height = fig[1, 1] = Axis(fig)
    heatmap!(ax_dist_height, CoordDivs[:x], CoordDivs[:height], depth_vs_height)
    ax_dist_height.xlabel = "Depth"
    ax_dist_height.ylabel = "Height"
    display(fig)
    return mh_traces, accepted_list, xvals
end


function grid_enumerate(gen_fn, args, addr_valueset_pairs, input_tr)
    # addr_vaslueset_pairs = [(addr1, set_of_values_for_this_addr), (addr2, set), ...]
    logweights = Dict()
    addrs = [d[1] for d in addr_valueset_pairs]
    cmap_w_image2D = make_2D_constraints(input_tr)
    for assmt in Iterators.product((set for (_, set) in addr_valueset_pairs)...)
        [cmap_w_image2D[addr] = val for (addr, val) in zip(addrs, assmt)]
        _, weight = Gen.generate(gen_fn, args, cmap_w_image2D)
        logweights[assmt] = weight
    end
    return logweights, addrs
end
# want grid to be plotted as a cone.


function plot_grid_enumeration_weights(logweights, addrs, addr_x, addr_y)

    addr_x_index = findfirst(f -> f == addr_x, addrs)
    addr_y_index = findfirst(f -> f == addr_y, addrs)
    
    prob_matrix = zeros(length(CoordDivs[addr_x]), length(CoordDivs[addr_y]))
    for (xind, xval) in enumerate(CoordDivs[addr_x]), (yind, yval) in enumerate(CoordDivs[addr_y])
        weights_for_gridcell = []
        for key in keys(logweights)
            if key[addr_x_index] == xval && key[addr_y_index] == yval
                push!(weights_for_gridcell, logweights[key])
            end
        end
        if  logsumexp(convert(Array{Float64, 1}, weights_for_gridcell)) != -Inf
            println(xval)
            println(yval)
            println(logsumexp(convert(Array{Float64, 1}, weights_for_gridcell)))
        end
        prob_matrix[xind, yind] = exp(logsumexp(convert(Array{Float64, 1}, weights_for_gridcell)))
    end
    fig = Figure()
    ax = fig[1, 1] = Axis(fig)
    heatmap!(ax, CoordDivs[addr_x], CoordDivs[addr_y], prob_matrix, colormap=:thermal)
    ax.xlabel = string(addr_x)
    ax.ylabel = string(addr_y)
    display(fig)
    return ax
end




# OK! seems like this all works in principle. figure out what you want your inference results to feel like tomorrow



        
# only need this to go back to an XYZ world location. this is for SLAM or something like that; i.e. an allocentric position
# in xyz world space
        
#function project_to_xyz(trace::, cam::Detector)
    
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
            (CoordDivs[:az][az_i], CoordDivs[:alt][alt_i], CoordDivs[:r][dist_i])),
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
    projected_vals = [{(az_i, alt_i)} ~ recur_light_projection(az_i, alt_i, length(CoordDivs[:r]), world_state, 0)
                      for az_i=1:length(CoordDivs[:az]),
                          alt_i=1:length(CoordDivs[:alt])]
    return projected_vals
end

    

# THIS IS FOR TRANSFORMING TO A CAMERA WITH A YAW AND PITCH
# z_in_basis = r * sin(alt)
# y_in_basis = r * cos(alt) * cos(az)
# x_in_basis = r * cos(alt * sin(az)

# slightly more complex version of linefinding

# function vertical_line_finder(image2D)
#     nonzero_azalt = sort(findall(f -> f > 0, image2D))
#     object_dict = Dict()
#     [object_dict[i] = [] for i in SphericalTiles[:az]]
#     [push!(object_dict[ci[1]], ci[2]) for ci in nonzero_azalt]
#     line_cands = []
#     for (az_div, az_range) in enumerate(SphericalTiles)
#         alt_coords = object_dict[az_div]
#         if length(alt_coords) > 2
# end
