
const MLJManifoldPoint = Tuple{Any,Manifold}

"""
    TangentSpaceTransformer(p = :mean)

Unsupervised model for transforming data to tangent space.
`p` is either equal to `:mean`, in which case data will be transformed to the tangent
space at the mean of points given to `fit`, or a specific point on a manifold (represented as a tuple).
"""
mutable struct TangentSpaceTransformer <: Unsupervised
    p::Union{Symbol,MLJManifoldPoint}
end

TangentSpaceTransformer() = TangentSpaceTransformer(:mean)

function MLJBase.fit(transformer::TangentSpaceTransformer, verbosity::Int, v::AbstractVector{<:MLJManifoldPoint})
    M = v[1][2]
    if transformer.p === :mean
        point = (mean(M, map(p -> p[1], v)), M)
    else
        point = transformer.p
    end
    fitresult = (point..., DefaultOrthonormalBasis())
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end


function MLJBase.fitted_params(::TangentSpaceTransformer, fitresult)
    return (point = fitresult[1:2], basis = fitresult[3])
end

# for transforming single value:
function MLJBase.transform(transformer::TangentSpaceTransformer, fitresult, p::MLJManifoldPoint)
    q, M, basis = fitresult
    X = log(M, q, p[1])
    coeffs = get_coefficients(M, q, X, basis)
    new_features = [Symbol("X_$i") for i in 1:manifold_dimension(M)]
    named_cols = NamedTuple{tuple(new_features...)}(tuple(coeffs)...)
    return MLJBase.table(named_cols)
end

# for transforming vector:
function MLJBase.transform(transformer::TangentSpaceTransformer, fitresult, ps)
    return [transform(transformer, fitresult, p) for p in ps]
end

# for single values:
function MLJBase.inverse_transform(transformer::TangentSpaceTransformer, fitresult, coeffs)
    q, M, basis = fitresult
    X = get_vector(M, q, coeffs, basis)
    p = exp(M, q, X)
    return (p, M)
end

# for vectors:
function MLJBase.inverse_transform(transformer::TangentSpaceTransformer, fitresult, w::AbstractVector)
    return [inverse_transform(transformer, fitresult, y) for y in w]
end
