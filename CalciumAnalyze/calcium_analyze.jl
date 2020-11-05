using Pkg
pkg"activate ."
using ComputationalResources
#using Libdl
#using ArrayFire
using Images
using Makie
using AbstractPlotting
using AbstractPlotting.MakieLayout
using GLMakie
using ImageFiltering
using Random
using Statistics
using ShiftedArrays
using PyCall
using LinearAlgebra
using LightGraphs
using MetaGraphs
using MultivariateStats

#brian2 = pyimport("brian2")
#tiffstack = load("/Users/nightcrawler2/Desktop/2P_analysis/Fish1_visstim_340_zoom2_single_dot_slow_0.tif");
tiffstack = load("/Volumes/Esc_and_2P/201027/Fish1_whole_tectum_lowerdot_0.tif")
ts_gray = Gray.(tiffstack);
im_res = size(ts_gray[:,:,1])[1]
kernel_width = 170
im_center = im_res ÷ 2
kernel_boundaries = im_center - kernel_width ÷ 2 : im_center + kernel_width ÷ 2
addresource(ArrayFireLibs)

# Note 2P values are a direct count of photons. There is a max
# cutoff at 10 -- so norm to 255 / 10 for display, but not analysis. 

# NEXT TWO STEPS
# 1. Choose 4 points using bookmarked makie mousepress on a max projected image. use to create
# rotated ellipse equation and function (i.e. if less than 1, inside ellipse). Once you've done this,
# filter out pixels for presence inside the ellipse. run them through the ROI program, but do not restrict the
# size of the ROI (i.e. the max_pixels variable). this way you can get a blanket of ROIs.

# 2. Want a PCA function that takes in ROI activities. Return the first 3 PCs and create an RGB color from them. 


"""Creates an average image of the entire stack"""

function project_stack(stack::Array{Gray{Normed{UInt8, 8}}, 3})
    ms_stack_sum = sum(stack, dims=3) .- mean(sum(stack, dims=3))
    max_stack, min_stack = findmax(ms_stack_sum)[1], findmin(ms_stack_sum)[1]
    normed_stack = (ms_stack_sum .- min_stack) ./ (max_stack - min_stack)
    convert_to_nof8 = convert(Array{Gray{Normed{UInt8, 8}}, 2}, normed_stack[:,:,1])
    return convert_to_nof8
end

"""Finds the maximum 2D correlation position of the kernel (summed stack) run over the image of interest"""

function crosscorr_images(img::Array{Gray{Normed{UInt8,8}},2}, template::Array{Gray{Float64}, 3})
    # note there is a call to the GPU you can make here. would speed up significantly off the laptop
#    z = imfilter(ArrayFireLibs(), img .- mean(img), centered(template[:,:,1] .- mean(template[:,:,1])), Inner(), Algorithm.FIR())
    z = imfilter(img .- mean(img), centered(template[:,:,1] .- mean(template[:,:,1])), Inner(), Algorithm.FIR())
    max_inds = findmax(z)
    return max_inds
end    

"""Recursively aligns each image in the stack using cross correlation"""


function motion_correct_xy(indices,
                           stack::Array{Gray{Normed{UInt8,8}},3})
    if isempty(indices)
        save("motion_corrected.tif", stack)
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

roi_outline(id, roi_trees) = [Tuple(mean([get_prop(roi_trees[id], v, :coord) for v in vertices(roi_trees[id])]))]

mean_node(id, roi_trees) = mean([[n.val for n in get_prop(roi_trees[id], v, :activity)] for v in vertices(roi_trees[id])])

spatial_filter(bright_pix, ellipse_func, λ_ellipse) = [bp for bp in bright_pix if λ_ellipse(ellipse_func(bp[1], bp[2]))]

# if you want inside ellipse, make λ_ellipse x -> x < 1, outside x -> x > 1

# PUT ALL THESE VARBS INTO A DICTIONARY AND PLOT A TRANSPARENT MASK OVER
# EACH ROI ON A PROJECTED STACK ACCORDING TO THE RGB VALS. 

function pca_on_traces(roi_trees::Array{MetaDiGraph{Int64, Float64}, 1},
                       stack::Array{Gray{Normed{UInt8,8}},3})
    scene = Scene();
    calcium_traces = [mean_node(id, roi_trees) for id in 1:size(roi_trees)[1]]
    ca_matrix = hcat(calcium_traces...)
    pca_m = fit(PCA, ca_matrix)
    first3PCs = projection(pca_m)[:, 1:3]
    first3Vars = principalvars(pca_m)[1:3]
    proj_of_traces = [transform(pca_m, c)[1:3] for c in calcium_traces]
    pca_dict = Dict("CaTraces" => calcium_traces,
                    "3PC" => first3PCs, "3Vars" => first3Vars, "Proj" => proj_of_traces);
    return pca_dict
end

function plot_pc(pc_dict::Dict,
                 pc_ind::Integer,
                 ca_trace_ind::Integer)
    pc = [convert(Tuple{Float32, Float32}, a) for a in enumerate(pc_dict["3PC"][:,pc_ind])]
    ca = [convert(Tuple{Float32, Float32}, a) for a in enumerate(pc_dict["CaTraces"][ca_trace_ind])]    
    s = Scene();
    scatter!(s, pc, strokecolor=:red)
    lines!(s, pc, strokecolor=:red)
    scatter!(s, ca)
    lines!(s, ca)
    return s 
end    

ellipse_bounds = []

function draw_roi(stack::Array{Gray{Normed{UInt8,8}},3})
    global ellipse_bounds = []
    prj = project_stack(stack)
    s = Scene(resolution=(800, 800))
    image!(s, prj)
    clicks = Node(Point2f0[])
    on(s.events.mousebuttons) do buttons
        if ispressed(s, Mouse.left)
            pos = to_world(s, Point2f0(s.events.mouseposition[]))
            clicks[] = push!(clicks[], pos)
            push!(ellipse_bounds, pos)
        end
        return
    end
    scatter!(s, clicks, color = :red, marker = '+', markersize = 20)
    RecordEvents(s, "output")
    s
end

function ROI_ellipse()
    e_bounds = [e for e in ellipse_bounds if e[1] > 0]
    ycoords = [e[2] for e in ellipse_bounds]
    xcoords = [e[1] for e in ellipse_bounds]
    h = (maximum(xcoords) - minimum(xcoords)) / 2
    k = (maximum(ycoords) - minimum(ycoords)) / 2
    rot_coords = e_bounds[findmax(ycoords)[2]] - [h, k]
    θ = atan(rot_coords[1] / rot_coords[2])
    a = norm(e_bounds[findmax(ycoords)[2]] - k)
    b = norm(e_bounds[findmax(xcoords)[2]] - h)
    f(x, y) = ((((x-h)*cos(θ) + (y-k)*sin(θ))^2) / a^2) +
        ((((x-h)*sin(θ) - (y-k)*cos(θ))^2) / b^2)
    return f
end    


    
function roi_activity_viewer(roi_trees::Array{MetaDiGraph{Int64, Float64}, 1},
                             stack::Array{Gray{Normed{UInt8,8}},3})
    #this function will pull average activity out of a metagraph
    scene, layout = layoutscene(backgroundcolor=RGBf0(255,255,255), resolution=(2000,1000));
    ncols = 2
    nrows = 1
    # create a grid of LAxis objects
    axes = [LAxis(scene, backgroundcolor=RGBf0(255, 255, 255)) for i in 1:nrows, j in 1:ncols]
    axes[1].ylabel = "Photon Count"
    layout[1:nrows, 1:ncols] = axes
    s1 = slider!(scene, range(1, stop=size(stack)[3]), raw=true, camera=campixel!, start=size(stack)[3])
    roi_id = 1
    max_fluorval = maximum([maximum(mean_node(i, roi_trees)) for i in 1:size(roi_trees)[1]])
    current_roi = lift(scene.events.keyboardbuttons) do but
        if ispressed(but, Keyboard.left)
            return roi_id -= 1
        elseif ispressed(but, Keyboard.right)
            return roi_id += 1
        else
            return roi_id
        end
    end
    brainslice = lift(s1[end][:value]) do v
        println(current_roi)
        stack[:, :, convert(Int64, v)] * (255 / 10)
    end
    lines!(axes[1], lift(x-> mean_node(x, roi_trees), current_roi), backgroundcolor=:black)
    limits!(axes[1], BBox(0, size(stack)[3], 0, max_fluorval))
    image!(axes[2], brainslice)
    scatter!(axes[2], lift(x-> roi_outline(x, roi_trees), current_roi), 
             color=:transparent, strokecolor=:red, strokewidth=3, markersize=25)
    display(scene)
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
                                                 :activity => stack[new_vertex_coord[1],
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
                           stack::Array{Gray{Normed{UInt8,8}},3},
                           max_pixels::Integer)
    seed = rand(bright_pixels)
    add_vertex!(roi_tree)
    set_props!(roi_tree, nv(roi_tree), Dict(:coord => seed, 
                                            :activity => stack[seed[1],
                                                               seed[2],:]))
    generate_roi_tree([seed],
                      filter(x -> x!=seed, bright_pixels),
                      roi_tree, stack, max_pixels)
end    

# problem that there's no return here? should there be a catch for no bright pixels? i think.         
function generate_roi_tree(seeds::Array{Array{Int64, 1}},
                           bright_pixels::Array{Array{Int64, 1}},
                           roi_tree::MetaDiGraph{Int64, Float64}, 
                           stack::Array{Gray{Normed{UInt8,8}},3},
                           max_pixels::Integer)
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
                              roi_tree, stack, max_pixels)
        end
    else
        println("neighbors detected")
        roi_tree_update = add_edges_to_seed(seed, neighbors, roi_tree, stack)
        if nv(roi_tree_update) > max_pixels
            return filter(x -> x!=seed, bright_pixels), roi_tree_update
        end
        generate_roi_tree(neighbors,
                          filter(x -> x!=seed && !(x in neighbors), bright_pixels),
                          roi_tree, stack, max_pixels)
    end
end    
                           
function make_graph_list(bright_pixels::Array{Array{Int64, 1}},
                         stack::Array{Gray{Normed{UInt8,8}},3},
                         max_pixels::Integer)
    graph_list = []
    while !isempty(bright_pixels)
        bright_pixels, roi_tree = generate_roi_tree(bright_pixels, MetaDiGraph(), stack, max_pixels)
        push!(graph_list, roi_tree)
    end
    return graph_list
end        


#aligned_stack = motion_correct_xy(range(1, stop=size(ts_gray)[3]), ts_gray);
aligned_stack = load("motion_corrected.tif");
graph_list = make_graph_list(get_bright_pixels(aligned_stack, .2), aligned_stack, 30);

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

