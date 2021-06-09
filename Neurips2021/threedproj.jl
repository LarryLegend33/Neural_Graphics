using GLMakie
using Gen

struct SphericalVoxel
    az::Float64
    alt::Float64
    r::Int64
    alpha::Float64
    brightness::Float64
    # color::
end

struct Object3D
    voxels::Array{SphericalVoxel}
    # eventually want to give each voxel material a color
end


spherical_overlap(vox::SphericalVoxel, coord::Tuple) = coord[1] == vox.az && coord[2] == vox.alt && coord[3] == vox.r

const CoordDivs = Dict(:az => collect(-50:5:50), 
                        :alt => collect(-50:5:50),
                        :dist => collect(0:20))

photon_count_probs(n, max) = .5 ^(n) * [i <= n ? binomial(n, i) : 0. for i=0:max]
onehot(v, dom) = [x == v ? 1 : 0 for x in dom]
@dist LabeledCat(labels, pvec) = labels[categorical(pvec)]

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
            # neural noise encoded as an cognitive contribution of a voxel, which is the representation
            # here if transparency is 0, want to multiply by ~ 1.
            # if its near 1, want to multiply it by ~ 0.
            # note all filters are multiplicative and not additive.
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
    

@gen function produce_retinal_image(world_state::Array{SphericalVoxel})
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
    return convert(Array{SphericalVoxel}, world_state)
end

                                                     
        
    
    
    


