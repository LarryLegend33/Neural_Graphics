using Makie
using Gen
using LinearAlgebra


"""Create Makie Rendering Environment"""







"""Node in a tree representing a covariance function"""
abstract type Kernel end
abstract type PrimitiveKernel <: Kernel end
abstract type CompositeKernel <: Kernel end

"""Number of nodes in the tree describing this kernel."""
Base.size(::PrimitiveKernel) = 1
Base.size(node::CompositeKernel) = node.size

"""Constant kernel"""
struct Constant <: PrimitiveKernel
    param::Float64
end

eval_cov(node::Constant, x1, x2) = node.param

function eval_cov_mat(node::Constant, xs::Vector{Float64})
    n = length(xs)
    fill(node.param, (n, n))
end


"""Linear kernel"""
struct Linear <: PrimitiveKernel
    param::Float64
end

eval_cov(node::Linear, x1, x2) = (x1 - node.param) * (x2 - node.param)

function eval_cov_mat(node::Linear, xs::Vector{Float64})
    xs_minus_param = xs .- node.param
    xs_minus_param * xs_minus_param'
end


"""Squared exponential kernel"""
struct SquaredExponential <: PrimitiveKernel
    length_scale::Float64
end

eval_cov(node::SquaredExponential, x1, x2) =
    exp(-0.5 * (x1 - x2) * (x1 - x2) / node.length_scale)

function eval_cov_mat(node::SquaredExponential, xs::Vector{Float64})
    diff = xs .- xs'
    exp.(-0.5 .* diff .* diff ./ node.length_scale)
end


"""Periodic kernel"""
struct Periodic <: PrimitiveKernel
    scale::Float64
    period::Float64
end

function eval_cov(node::Periodic, x1, x2)
    freq = 2 * pi / node.period
    exp((-1/node.scale) * (sin(freq * abs(x1 - x2)))^2)
end

function eval_cov_mat(node::Periodic, xs::Vector{Float64})
    freq = 2 * pi / node.period
    abs_diff = abs.(xs .- xs')
    exp.((-1/node.scale) .* (sin.(freq .* abs_diff)).^2)
end


"""Plus node"""
struct Plus <: CompositeKernel
    left::Kernel
    right::Kernel
    size::Int
end

Plus(left, right) = Plus(left, right, size(left) + size(right) + 1)

function eval_cov(node::Plus, x1, x2)
    eval_cov(node.left, x1, x2) + eval_cov(node.right, x1, x2)
end

function eval_cov_mat(node::Plus, xs::Vector{Float64})
    eval_cov_mat(node.left, xs) .+ eval_cov_mat(node.right, xs)
end


"""Times node"""
struct Times <: CompositeKernel
    left::Kernel
    right::Kernel
    size::Int
end

Times(left, right) = Times(left, right, size(left) + size(right) + 1)

function eval_cov(node::Times, x1, x2)
    eval_cov(node.left, x1, x2) * eval_cov(node.right, x1, x2)
end

function eval_cov_mat(node::Times, xs::Vector{Float64})
    eval_cov_mat(node.left, xs) .* eval_cov_mat(node.right, xs)
end


"""Compute covariance matrix by evaluating function on each pair of inputs."""
function compute_cov_matrix(covariance_fn::Kernel, noise, xs)
    n = length(xs)
    cov_matrix = Matrix{Float64}(undef, n, n)
    for i=1:n
        for j=1:n
            cov_matrix[i, j] = eval_cov(covariance_fn, xs[i], xs[j])
        end
        cov_matrix[i, i] += noise
    end
    return cov_matrix
end


"""Compute covariance function by recursively computing covariance matrices."""
function compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    n = length(xs)
    eval_cov_mat(covariance_fn, xs) + Matrix(noise * LinearAlgebra.I, n, n)
end

"""
Computes the conditional mean and covariance of a Gaussian process with prior mean zero
and prior covariance function `covariance_fn`, conditioned on noisy observations
`Normal(f(xs), noise * I) = ys`, evaluated at the points `new_xs`.
"""
function compute_predictive(covariance_fn::Kernel, noise::Float64,
                            xs::Vector{Float64}, ys::Vector{Float64},
                            new_xs::Vector{Float64})
    n_prev = length(xs)
    n_new = length(new_xs)
    means = zeros(n_prev + n_new)
    cov_matrix = compute_cov_matrix(covariance_fn, noise, vcat(xs, new_xs))
    cov_matrix_11 = cov_matrix[1:n_prev, 1:n_prev]
    cov_matrix_22 = cov_matrix[n_prev+1:n_prev+n_new, n_prev+1:n_prev+n_new]
    cov_matrix_12 = cov_matrix[1:n_prev, n_prev+1:n_prev+n_new]
    cov_matrix_21 = cov_matrix[n_prev+1:n_prev+n_new, 1:n_prev]
    @assert cov_matrix_12 == cov_matrix_21'
    mu1 = means[1:n_prev]
    mu2 = means[n_prev+1:n_prev+n_new]
    conditional_mu = mu2 + cov_matrix_21 * (cov_matrix_11 \ (ys - mu1))
    conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 * (cov_matrix_11 \ cov_matrix_12)
    conditional_cov_matrix = 0.5 * conditional_cov_matrix + 0.5 * conditional_cov_matrix'
    (conditional_mu, conditional_cov_matrix)
end

"""
Predict output values for some new input values
"""
function predict_ys(covariance_fn::Kernel, noise::Float64,
                    xs::Vector{Float64}, ys::Vector{Float64},
                    new_xs::Vector{Float64})
    (conditional_mu, conditional_cov_matrix) = compute_predictive(
        covariance_fn, noise, xs, ys, new_xs)
    mvnormal(conditional_mu, conditional_cov_matrix)
end

kernel_types = [Constant, Linear, SquaredExponential, Periodic, Plus, Times]
@dist choose_kernel_type() = kernel_types[categorical([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])];


# Prior on kernels
@gen function covariance_prior()
    # Choose a type of kernel
    kernel_type ~ choose_kernel_type()

    # If this is a composite node, recursively generate subtrees
    if in(kernel_type, [Plus, Times])
        return kernel_type({ :left } ~ covariance_prior(), { :right } ~ covariance_prior())
    end
    
    # Otherwise, generate parameters for the primitive kernel.
    kernel_args = (kernel_type == Periodic) ? [{:scale} ~ uniform(0, 1), {:period} ~ uniform(0, 1)] : [{:param} ~ uniform(0, 1)]
    return kernel_type(kernel_args...)
end


@dist gamma_bounded_below(shape, scale, bound) = gamma(shape, scale) + bound

# Full model
@gen function cv_generation_model(xs::Vector{Float64})
    
    # Generate a covariance kernel
    covariance_fn = { :tree } ~ covariance_prior()
    
    # Sample a global noise level
    noise ~ gamma_bounded_below(1, 1, 0.01)
    
    # Compute the covariance between every pair (xs[i], xs[j])
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    
    # Sample from the GP using a multivariate normal distribution with
    # the kernel-derived covariance matrix.
    ys ~ mvnormal(zeros(length(xs)), cov_matrix)
    
    # Return the covariance function, for easy printing.
    return covariance_fn
end;


function serialize_trace(tr, xmin, xmax)
    (xs,) = get_args(tr)
    curveXs = collect(Float64, range(xmin, length=100, stop=xmax))
    curveYs = [predict_ys(get_retval(tr), 0.00001, xs, tr[:ys],curveXs) for i=1:5]
    Dict("y-coords" => tr[:ys], "curveXs" => curveXs, "curveYs" => curveYs)
end

for iter=1:20
    (tr, _) = generate(cv_generation_model, (collect(Float64, -1:0.1:1),))
    serialize_trace(tr, -5, 5)
end


struct XYInit
    x::Float64
    y::Float64
end

struct VelocityVector
    x::Float64
    y::Float64
end

abstract type Node end

struct InternalNode <: Node
    left::Node
    right::Node
    xy::XYInit
    vvector::VelocityVector
end

struct LeafNode <: Node
    xy::XYInit
    vvector::VelocityVector
end

@gen function generate_dotmotion()
    if ({ :isleaf } ~ bernoulli(0.7))
        value = ({ :value } ~ normal(0, 1))
        #generate a covariance function and velocities here
        return LeafNode(value, interval)
    else
        frac = ({:frac} ~ beta(2, 2))
        mid  = l + (u - l) * frac
        # Call generate_segments recursively!
        # Because we will call it twice -- one for the left 
        # child and one for the right child -- we use
        # addresses to distinguish the calls.
        left = ({:left} ~ generate_segments(l, mid))
        right = ({:right} ~ generate_segments(mid, u))
        return InternalNode(left, right, interval)
    end
end;
