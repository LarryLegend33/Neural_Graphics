using Images
using Makie
using AbstractPlotting
using AbstractPlotting.MakieLayout
using GLMakie
using ImageFiltering
using Random
using Statistics
using ShiftedArrays
#using PyCall
using LinearAlgebra
using LightGraphs
using MetaGraphs


#brian2 = pyimport("brian2")
tiffstack = load("/Users/nightcrawler2/Desktop/2P_analysis/Fish1_visstim_340_zoom2_single_dot_slow_0.tif");
ts_gray = Gray.(tiffstack);
im_res = size(ts_gray[:,:,1])[1]
kernel_width = 100
im_center = im_res ÷ 2
kernel_boundaries = im_center - kernel_width ÷ 2 : im_center + kernel_width ÷ 2

# Note 2P values are a direct count of photons. There is a max
# cutoff at 10 -- so norm to 255 / 10 for display, but not analysis. 

outer_padding = 0
scene, layout = layoutscene(outer_padding, resolution = (im_res, im_res),
                            backgroundcolor=RGBf0(0,0,0))



"""Creates an average image of the entire stack"""

function project_stack(stack::Array{Gray{Normed{UInt8, 8}}, 3})
    ms_stack_sum = sum(stack, dims=3) .- mean(sum(stack, dims=3))
    max_stack, min_stack = findmax(ms_stack_sum)[1], findmin(ms_stack_sum)[1]
    normed_stack = (ms_stack_sum .- min_stack) ./ (max_stack - min_stack)
    convert_to_nof8 = convert(Array{Gray{Normed{UInt8, 8}}, 2}, normed_stack[:,:,1])
    image!(scene, convert_to_nof8)
    display(scene)
    return convert_to_nof8
end

"""Finds the maximum 2D correlation position of the kernel (summed stack) run over the image of interest"""

function crosscorr_images(img::Array{Gray{Normed{UInt8,8}},2}, template::Array{Gray{Float64}, 3})
    # note there is a call to the GPU you can make here. would speed up significantly off the laptop
    z = imfilter(img .- mean(img), centered(template[:,:,1] .- mean(template[:,:,1])), Inner(), Algorithm.FIR())
    max_inds = findmax(z)
    return max_inds
end    



"""Recursively aligns each image in the stack using cross correlation"""

function motion_correct_xy(indices,
                           stack::Array{Gray{Normed{UInt8,8}},3})
    if isempty(indices)
        return stack
    end
    id_to_align = rand(indices)
    image_to_align = stack[:, :, id_to_align]
    max_corr_coords = crosscorr_images(image_to_align,
                                       sum(stack[kernel_boundaries, kernel_boundaries, 1:end .!=id_to_align], dims=3))
    shift_coords = convert(Tuple, max_corr_coords[2] - CartesianIndex(im_res÷2, im_res÷2))
    aligned_image = ShiftedArray(image_to_align, shift_coords, default=Gray{N0f8}(0.0))
    motion_correct_xy(filter(x -> x!=id_to_align, indices), 
                      cat(stack[:, :, 1:id_to_align-1],
                          aligned_image,
                          stack[:, :, id_to_align+1:end], dims=3))
end

"""Retrieves all pixel coordinates"""

get_bright_pixels(stack, brightness_thresh) = [[x, y] for x in range(1, stop=size(stack)[1]),
                                               y in range(1, stop=size(stack)[2]) if sum(stack[x, y, :]) > brightness_thresh]

# are going to add seed to a function. has to check the neighbors

# use size(find_longest_path(g).longest_path)[1]

function roi_activity(roi_tree::MetaDiGraph{Int64, Float64})
    #this function will pull average activity out of a metagraph
    a=5
end


function add_edges_to_seed(seed::Array{Int64, 1},
                           neighbors::Array{Array{Int64, 1}},
                           roi_tree::MetaDiGraph{Int64, Float64},
                           stack::Array{Gray{Normed{UInt8,8}},3})
    if isempty(neighbors)
        return roi_tree
    else
        new_vertex_coord = neighbors[1]
        add_vertex!(roi_tree)
        new_vertex_id = nv(roi_tree)
        set_props!(roi_tree, new_vertex_id, Dict(:coord => new_vertex_coord,
                                                 :timeseries => stack[new_vertex_coord[1],
                                                                      new_vertex_coord[2],:]))
        add_edge!(roi_tree,
                  filter(i -> get_prop(roi_tree, i, :coord) == seed,
                         vertices(roi_tree))[1],
                  new_vertex_id)
        add_edges_to_seed(seed, neighbors[2:end], roi_tree, stack)
    end
end    


        
#Initialize this with empty list, brightpixels, empty graph, stack
# a graph can also end if ALL neighbors are empty 

function generate_roi_tree(bright_pixels::Array{Array{Int64, 1}},
                           roi_tree::MetaDiGraph{Int64, Float64}, 
                           stack::Array{Gray{Normed{UInt8,8}},3})
    seed = rand(bright_pixels)
    add_vertex!(roi_tree)
    set_props!(roi_tree, nv(roi_tree), Dict(:coord => seed, 
                                            :timeseries => stack[seed[1],
                                                                 seed[2],:]))
    generate_roi_tree([seed],
                      filter(x -> x!=seed, bright_pixels),
                      roi_tree, stack)
end    

# problem that there's no return here? should there be a catch for no bright pixels? i think.         
function generate_roi_tree(seeds::Array{Array{Int64, 1}},
                           bright_pixels::Array{Array{Int64, 1}},
                           roi_tree::MetaDiGraph{Int64, Float64}, 
                           stack::Array{Gray{Normed{UInt8,8}},3})
    println(size(bright_pixels))
    seed = seeds[1]
    corr_thresh = .6
    neighbors = filter(
        c -> !(c in seeds) && abs(
            c[1] - seed[1]) <= 1 && abs(
                c[2]-seed[2]) <= 1 && cor(stack[seed[1], seed[2], :],
                                          stack[c[1], c[2], :]) > corr_thresh,
        bright_pixels)

    if isempty(neighbors)
        # this scenario hits if a random seed has no partners, or if the neighbor pass is over. 
        if size(seeds)[1] == 1
            return filter(x -> x!=seed, bright_pixels), roi_tree
        else
            generate_roi_tree(seeds[2:end],
                              filter(x -> x!=seed, bright_pixels),
                              roi_tree, stack)
        end
    else
        println("neighbors detected")
        roi_tree_update = add_edges_to_seed(seed, neighbors, roi_tree, stack)
        if nv(roi_tree_update) > 30
            return filter(x -> x!=seed, bright_pixels), roi_tree_update
        end
        generate_roi_tree(neighbors,
                          filter(x -> x!=seed && !(x in neighbors), bright_pixels),
                          roi_tree, stack)
    end
end    
                           
function make_graph_list(bright_pixels::Array{Array{Int64, 1}},
                         stack::Array{Gray{Normed{UInt8,8}},3})
    graph_list = []
    while !isempty(bright_pixels)
        bright_pixels, roi_tree = generate_roi_tree(bright_pixels, MetaDiGraph(), stack)
        push!(graph_list, roi_tree)
    end
    return graph_list
end        
        


#aligned_stack = motion_correct_xy(range(1, stop=size(ts_gray)[3]),
#                                  ts_gray);

graph_list = make_graph_list(get_bright_pixels(aligned_stack, 4), aligned_stack);

# there's no tail call optimization in julia! so be careful how many max times you want the function to be called.
# slows down significantly after ~1000 calls. instead, return a graph and a pixel list after each seed. 



    





#= XY ALIGN STACK first. Take each image in the stack and align it to the 
sum of the stack without that image. You do this by filter2D of the stack sum with 
the image itself. This yields a cross correlation matrix where the image is a moving
window over the stack. If it's properly aligned, the position where the filter 
best matches the image will be at the center (i.e. the middle position of center iteration over 
the image starting in the top left corner and ending at the bottom right). The shift
for a particular image is the max of the cross correlation matrix minus the center. 
Want to do the alignment randomly and replace the sum's extracted image-of-interest as you go. =#

