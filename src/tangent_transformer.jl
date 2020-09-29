
const MLJManifoldPoint = Tuple{Any,Manifold}

"""
    TangentSpaceTransformer(p = :mean)

Unsupervised model for transforming data to tangent space.
`p` is either equal to `:mean`, in which case data will be transformed to the tangent
space at the mean of points given to `fit`, or a specific point on a manifold (represented as a tuple).
"""
mutable struct TangentSpaceTransformer <: Unsupervised
    p::Union{Symbol,MLJManifoldPoint}
    features::AbstractVector{Symbol} # if not empty only these features will be transformed
end

TangentSpaceTransformer() = TangentSpaceTransformer(:mean)

function univariate_to_tspace(M::Manifold, p, v)
    if p === :mean
        point = (mean(M, map(q -> q[1], v)), M)
    else
        point = p
    end
    fitresult = (point..., DefaultOrthonormalBasis())
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end

function MLJBase.fit(transformer::TangentSpaceTransformer, verbosity::Int, X)
    is_univariate = !Tables.istable(X)

    if is_univariate
        M = v[1][2]
        return (is_univariate = true, fitresult = univariate_to_tspace(M, transformer.p, X)), nothing, nothing
    end

    all_features = Tables.schema(X).names
    feature_scitypes = collect(elscitype(selectcols(X, c)) for c in all_features)
    if isempty(transformer.features)
        cols_to_fit = filter!(eachindex(all_features) |> collect) do j
            return feature_scitypes[j] <: ManifoldPoint
        end
    else
        error("TODO")
    end
    fitresult_given_feature = Dict{Symbol,Any}()

    isempty(cols_to_fit) && verbosity > -1 && @warn "No features to transform"
    for j in cols_to_fit
        col_data = selectcols(X, j)
        col_fitresult, cache, report = univariate_to_tspace(col_data[1][2], transformer.p, col_data)
        fitresult_given_feature[all_features[j]] = col_fitresult
    end

    fitresult = (is_univariate = false, fitresult_given_feature = fitresult_given_feature)

    return fitresult, nothing, nothing
end


function MLJBase.fitted_params(::TangentSpaceTransformer, fitresult)
    if fitresult.is_univariate
        return (point = fitresult.fitresult[1:2], basis = fitresult.fitresult[3])
    else
        error("TODO")
    end
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
