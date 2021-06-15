using GLMakie
using Gen
using GenGridEnumeration
using LinearAlgebra
using ImageFiltering
using OrderedCollections
using Distributions


# eventually want to find a way to subtype the voxel types to define
# voxel-wide fields,
# repeat definition of fields and to define functions on Voxels (i.e. adding, preventing
# collision)

# ALSO note the illusion probably comes from the floor perspective.
# play with the camera location, ideas about perspective. things that are further will be
# further from the ground.



# start the recursion with dist_i = length(distance_divs)


# TODO

# 
# 1) debug the enumeration with more than two variables. start by using a discrete gaussian blur
# 2) write a lookup table for xyz -> az alt d transformation w/ a maybe 1 off. 
# 2) finish johanssen stimulus inference
# can write a voxel in XYZ that is smaller than the smallest az, alt, r grid tile
# 


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




struct Object3D
    voxels::Array{Voxel}
    # eventually want to give each voxel material a color
end

#spherical_overlap(vox::SphericalVoxel, coord::Tuple) = coord[1] == vox.az && coord[2] == vox.alt && coord[3] == vox.r

spherical_overlap(vox::SphericalVoxel,
                  tile) = within(vox.az, tile[1], SphericalTiles[:az][end][end]) && within(
                      vox.alt, tile[2], SphericalTiles[:alt][end][end]) && within(
                          vox.r, tile[3], SphericalTiles[:dist][end][end])

CoordDivs = Dict(:az => collect(-80.0:10.0:80.0), 
                 :alt => collect(-80:10:80),
                 :x => collect(2:2:20), 
                 :y => collect(-10:2:10),
                 :z => collect(-10:2:10),
                 :height => collect(2:20),
                 :brightness => collect(100:100))

CoordDivs[:dist] = collect(0:3:ceil(maximum([norm([x,y,z]) for x in CoordDivs[:x],
                                                 y in CoordDivs[:y],
                                                 z in CoordDivs[:z]])))
                           
SphericalTiles = Dict(:az => mapwindow(collect, CoordDivs[:az], 0:1, border=Inner()),
                      :alt => mapwindow(collect, CoordDivs[:alt], 0:1, border=Inner()),
                      :dist => mapwindow(collect, CoordDivs[:dist], 0:1, border=Inner()))

# only value that isn't caught by boundaries is the last member of the range.
# if the value is the last member of the domain and the last tile is the boundary. 
within(val, boundaries, b_max) = (val >= boundaries[1] && val < boundaries[end]) || (val == b_max && val == boundaries[end])
@dist LabeledCat(labels, pvec) = labels[categorical(pvec)]
@dist uniformLabCat(labels) = LabeledCat(labels, 1/length(labels) * ones(length(labels)))
findnearest(input_arr::AbstractArray, val) = input_arr[findmin([abs(a) for a in input_arr.-val])[2]]

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






# OK so now you have to make a model that generates objects in XYZ space and makes an
# azimuth altitude representation of them. so you generate an XYZ object,
# pass that to the spherical transform, then project that onto the retina.
# your latent states will contain a distance, az and alt, but the likelihood is performed
# on the az alt overlap. then write a program like move_step from energy model that moves the objects

# maybe you actually want to just start with a particular xyz coord for now.

# note you may actually want to infer photon counts behind occluders sometimes; depends on if you want to
# infer "brightnesses" or "objects"

# x is into page. y is horizontal. z is vertical. 

@gen function generate_image()
    # eventually draw more complex shapes
    shape = LabeledCat([:rect], [1])
    shape_height = { :shape_height } ~  uniformLabCat(CoordDivs[:height])
       
    brightness = { :brightness } ~ uniformLabCat(CoordDivs[:brightness])
    alpha = { :alpha } ~ uniform_discrete(1, 1)
    # this will eventually be the object's center, with the voxels defined by nishad's descriptions. 
    shape_x = { :shape_x } ~ uniformLabCat(CoordDivs[:x])
    shape_y = { :shape_y } ~ uniformLabCat(CoordDivs[:y])
    shape_z = { :shape_z } ~ uniformLabCat(CoordDivs[:z])
    origin_to_objectcenter_dist = Int64(floor(norm([shape_x, shape_y, shape_z+shape_height/2])))
    distance = { :r } ~ #Gen.normal(norm([shape_x, shape_y, shape_z+shape_height/2]), .01)
        uniform_discrete(origin_to_objectcenter_dist, origin_to_objectcenter_dist)
    # FIXTHIS -- this is just the middle of the object. want distance to each object component
    object_vox = []
    if shape == :rect
        for (i, z) in enumerate(shape_z:shape_z+shape_height)
            vox = XYZVoxel(shape_x, shape_y, z, alpha, brightness)
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
    az_alt_retina = { :image_2D } ~ produce_noisy_2D_retinal_image(set_world_state(object_in_spherical))
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
            [SphericalTiles[:az][az_i], SphericalTiles[:alt][alt_i], SphericalTiles[:dist][dist_i]]),
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


""" NOISE MODEL 2: GAUSSIAN ON A BOXFILTER """

# discretized normal distribution here. 

@gen function gaussian_noisegen(μ::Float64, maxpix::Float64)
    σ = 20.0
    pixrange = collect(0.0:maxpix)
    pix ~ LabeledCat(pixrange, discretized_gaussian(μ, σ, pixrange))
    #    pix ~ normal(μ, σ)
    
end

@gen function generate_blur(im_mat::Matrix{Float64}, filter_size::Int)
    # if all 9 are white pixels, still have a .1 chance of going black.
    # this will be offset by the baseline noise the other way. 
    conv_filter = ones(filter_size, filter_size) / (filter_size^2)
    #    maxpix = maximum(im_mat)
    maxpix = reduce(+ , [CoordDivs[:brightness][end] for d in SphericalTiles[:dist]])
    size_mat = size(im_mat)
    image_2D = { :image_2D } ~ Gen.Map(gaussian_noisegen)(
        imfilter(im_mat, conv_filter), maxpix*ones(size_mat[1]*size_mat[2]))
end
# it's firm here on no args to the Gen function. dont know how to get it there otherwise. 



@gen function produce_noisy_2D_retinal_image(world_state::Array{Voxel})
    projected_az_alt_vals = [recur_lightproj_deterministic(az_i, alt_i, length(SphericalTiles[:dist]), world_state, 0) for az_i=1:length(SphericalTiles[:az]), alt_i=1:length(SphericalTiles[:alt])]
    az_alt_mat = reshape(projected_az_alt_vals, (length(SphericalTiles[:az]), length(SphericalTiles[:alt])))
    noisy_projection ~ generate_blur(az_alt_mat, 1)
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
    println(tr[:shape_x])
    println("Y")
    println(tr[:shape_y])
    println("Z")
    println(tr[:shape_z])
    println("Height")
    println(tr[:shape_height])
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
    println(tr_populated[:r])
    println(tr_populated[:shape_height])
    tr = make_2D_constraints(tr_populated)
    for i in 1:amnt_computation
        (tr, accepted) = Gen.mh(tr, select(:shape_x, :shape_y, :shape_z, :r, :shape_height))
        push!(mh_traces, tr)
        push!(accepted_list, accepted)
    end

    # 
    dist_vs_height = zeros(length(CoordDivs[:dist]), length(CoordDivs[:height]))
    xvals = []
    for mht in mh_traces
        dist = mht[:r]
        height = tr[:shape_height]
        dist_vs_height[findfirst(d -> within(dist, d, SphericalTiles[:dist][end][end]),
                                 SphericalTiles[:dist]),
                       findfirst(h -> h == height, CoordDivs[:height])] += 1
        push!(xvals, tr[:shape_x])
    end
    fig = Figure()
    ax_dist_height = fig[1, 1] = Axis(fig)
    heatmap!(ax_dist_height, CoordDivs[:dist], CoordDivs[:height], dist_vs_height)
    ax_dist_height.xlabel = "Distance"
    ax_dist_height.ylabel = "Height"
    display(fig)
    return mh_traces, accepted_list, xvals
end


function enumeration_grid(input_trace::Gen.DynamicDSLTrace{DynamicDSLFunction{Any}})
    tr_w_constraints = make_2D_constraints(input_trace)
    g = UniformPointPushforwardGrid(tr_w_constraints, OrderedDict(
#        :shape_z => DiscreteSingletons(CoordDivs[:z]),
   #     :shape_y => DiscreteSingletons(CoordDivs[:y]),
        :shape_x => DiscreteSingletons(CoordDivs[:x]),
        :shape_height => DiscreteSingletons(CoordDivs[:height])))
    makie_plot_grid(g, :shape_x, :shape_height)
    println(input_trace[:shape_x])
    println(input_trace[:shape_height])
    return g
end



# want grid to be plotted as a cone. 

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
    # w is a matrix of weights ordered exactly how you'd expect. 
    w = let w_ = GenGridEnumeration.weights(g)
        addrs = collect(keys(g.addr2partition))
        (x_ind, y_ind) = indexin([x_addr, y_addr], addrs)
        dims = Tuple(setdiff(1:length(addrs), [x_ind, y_ind]))
        w = dropdims(sum(w_; dims=dims); dims=dims)
        x_ind < y_ind ? w : w'
    end
    # this is for plotting centers
    #    (x_heavy, y_heavy) = let (i_x, i_y) = Tuple(argmax(w))
    #        (repss[x_addr][i_x], repss[y_addr][i_y])
    #   end
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

    # definite problem here is you are just plotting the weights as a 3D matrix when it has to be a collapsed
    # 2D array. 
    
    heatmap!(ax, float(collect(sub_boundss[x_addr])),
             float(collect(sub_boundss[y_addr])), w, colormap=:thermal)
    display(f)
    return ax
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

    
