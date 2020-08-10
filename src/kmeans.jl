struct KMeansOptions{P} <: Options
    points::Vector{P}
    assignment::Vector{<:Int}
    centers::Vector{P}
    stop::StoppingCriterion
    function KMeansOptions{P}(points::Vector{P}, centers::Vector{P}, stop::StoppingCriterion) where {P}
        return new(points, zeros(length(points)), centers, stop)
    end
end

function KMeansOptions(points::Vector{P}, centers::Vector{P}, stop::StoppingCriterion=StopAfterIteration(100)) where {P}
    return KMeansOptions{P}(points, centers, stop)
end

struct KMeansProblem{TM <: Manifold} <: Problem
    M::TM
end

function initialize_solver(p::KMeansProblem, o::KMeansOptions)
    k_means_update_assignment!(p,o)
end

function solve!(p::KMeansProblem, o::KMeansOptions)
    # (1) Update assignments
    k_means_update_assignment!(p,o)
    # (2) Update centers
    for i=1:length(o.centers)
        any(o.assignment==i) && mean!(p.M, o.centers[i], o.points[o.assignment==i])
    end
end

function k_means_update_assignment!(p::KMeansProblem, o::KMeansOptions)
    for i=1:length(o.points)
        o.assignment[i] = argmin([ distance(p.M,o.points[i],c) for c in o.centers ] )
    end
end

"""
    kmeans(M,pts; numcenters=5, centers = pts[1:num_centers], stop=StopAfterIteration(100), kwargs)

Compute a simple k-means on a Riemannian manifold `M`.

Returns the final [`KMeansOptions`](@ref) including the final assignment vector.
"""
function kmeans(M::Manifold, pts::Vector{P};
    num_centers = 5,
    centers = pts[1:num_centers],
    stop=StopAfterIteration(100),
    kwargs...
    ) where {P}
    p = KMeansProblem(M)
    o =  KMeansOptions(pts,centers,stop)
    o = decorate_options(o; kwargs...)
    oR = solve(p,o)
    return get_options(oR)
end