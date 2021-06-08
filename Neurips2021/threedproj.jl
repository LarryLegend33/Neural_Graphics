using GLMakie
using Gen

begin
    Empty() = 1
    Opaque() = 2
    Translucent() = 3
    const TileState = Int
end

struct SphericalCoord
    az
    alt
    r
end

translucent_probs(n, max) = .5 ^(n) * [i <= n ? binomial(n, i) : 0. for i=0:max]
onehot(v, dom) = [x == v ? 1 : 0 for x in dom]
@dist LabeledCat(labels, pvec) = labels[categorical(pvec)]

@gen function produce_retinal_image(world_state::Array{TileState, 3},
                                    max_n_photons)
    return [{(:az, :alt)} ~ produce_light_value(world_state[az, alt, :], max_n_photons)
     for az=1:(size(world_state)[1]), alt=1:(size(world_state)[2])]
end


@gen function produce_light_value(tiles::Vector{TileState}, max_n_photons)
    n_light_particles = 0
    for (i, tile) in Iterators.reverse(enumerate(tiles))
        n_light_paricles = { :n => i } ~ LabeledCat(0:max_n_photons,
                                                    (if tile == Opaque()
                                                         onehot(max_n_photons, 0:max_n_photons)
                                                     elseif tile == Empty()
                                                         onehot(n_light_particles, 0:max_n_photons)
                                                     else
                                                         translucent_probs(n_light_particles, max_n_photons)
                                                     end))
    end
    return n_light_particles
end

function get_world_state(dims, opaques, translucents)
    world_state = [Empty() for az=1:dims[1], alt=1:dims[2], r=1:dims[3]]

    # this only works if the locations in the world are indexed instead of angles.
    # rewrite it so its in angles and the world state is a dictionary.
    

    
    world_state[map(SphericalCoord, opaques)] .= Opaque()
    world_state[map(SphericalCoord, translucents)] .= Translucent()
    return world_state
end

                                                     
        
    
    
    


