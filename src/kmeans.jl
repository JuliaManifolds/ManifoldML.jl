@doc """
    KMeansOptions <: Options

Collect the data necessary during computation of the k means clustering, i.e.

* `points::Vector{P}` – the given data
* `centers::Vector{P}` – the cluster centrs 
* `assignment::Vector{<:Int}` – a vector the same length as `points` assigning each of them to a cluster
* `stop::StoppingCriterion` a stoppingCriterion

Here `P` is a data type for points on the manifold the `points` (and `centers`) live on.
This manifold is stored in the [`KMeansProblem`](@ref).

# Constructor

    KMeansOptions(
        points::Vector{P},
        centers::Vector{P},
        stop::StoppingCriterion=StoppingCriterion(100)
    )

Initialize the options. The assignment is set to zero and initialized at the beginning of
the algorithm.
"""
struct KMeansOptions{P} <: Options
    points::Vector{P}
    centers::Vector{P}
    assignment::Vector{<:Int}
    stop::StoppingCriterion
    function KMeansOptions{P}(
        points::Vector{P},
        centers::Vector{P},
        stop::StoppingCriterion,
    ) where {P}
        return new(points, centers, zeros(Int, length(points)), stop)
    end
end
function KMeansOptions(
    points::Vector{P},
    centers::Vector{P},
    stop::StoppingCriterion = StopAfterIteration(100),
) where {P}
    return KMeansOptions{P}(points, centers, stop)
end

@doc """
    KMeansProblem <: Problem

Store the fixed data necessary for [`kmeans`](@ref), i.e. only a `Manifold M`.
"""
struct KMeansProblem{TM<:Manifold} <: Problem
    M::TM
end

function initialize_solver!(p::KMeansProblem, o::KMeansOptions)
    return k_means_update_assignment!(p, o)
end

function step_solver!(p::KMeansProblem, o::KMeansOptions, ::Int)
    # (1) Update assignments
    k_means_update_assignment!(p, o)
    # (2) Update centers
    for i in 1:length(o.centers)
        any(o.assignment == i) && mean!(p.M, o.centers[i], o.points[o.assignment == i])
    end
end

function k_means_update_assignment!(p::KMeansProblem, o::KMeansOptions)
    for i in 1:length(o.points)
        o.assignment[i] = argmin([distance(p.M, o.points[i], c) for c in o.centers])
    end
end

"""
    kmeans( M::Manifold, pts::Vector{P};
        num_centers=5,
        centers = pts[1:num_centers],
        stop=StopAfterIteration(100),
        kwargs...
    )

Compute a simple k-means on a Riemannian manifold `M` for the points `pts`.
The `num_centers` defaults to `5` and the initial centers `centers` are set to the first
`num_centers` data items. The stopping criterion is set by default to 100 iterations.

The `kwargs...` can be used to initialize [`RecordOptions`](https://manoptjl.org/stable/plans/index.html#RecordOptions-1) or [`DebugOptions`](https://manoptjl.org/stable/plans/index.html#DebugOptions-1)
decorators from [Manopt.jl](https://manoptjl.org)

Returns the final [`KMeansOptions`](@ref) including the final assignment vector and the centers.
"""
function kmeans(
    M::Manifold,
    pts::Vector{P};
    num_centers = 5,
    centers = pts[1:num_centers],
    stop = StopAfterIteration(100),
    kwargs...,
) where {P}
    p = KMeansProblem(M)
    o = KMeansOptions(pts, centers, stop)
    o = decorate_options(o; kwargs...)
    oR = solve(p, o)
    return get_options(oR)
end
