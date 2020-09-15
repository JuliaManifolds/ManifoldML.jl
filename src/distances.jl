
"""
    RiemannianDistance(M::Manifold)

An implementation of the `Metric` interface from the Distances.jl package that
returns Riemannian distance between points stored in a linearized array of coordinates.
"""
struct RiemannianDistance{TM<:Manifold} <: Metric
    manifold::TM
end

function (dist::RiemannianDistance)(a, b)
    rep_size = representation_size(dist.manifold)
    p = reshape(a, rep_size)
    q = reshape(b, rep_size)
    return distance(dist.manifold, p, q)
end
