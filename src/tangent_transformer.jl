
const MLJManifoldPoint = Tuple{Any,Manifold}


mutable struct UnivariateTangentSpaceTransformer <: Unsupervised
    p::Union{Symbol,MLJManifoldPoint}
    retraction::AbstractRetractionMethod
    inverse_retraction::AbstractInverseRetractionMethod
end

function UnivariateTangentSpaceTransformer()
    return UnivariateTangentSpaceTransformer(
        :mean,
        ExponentialRetraction(),
        LogarithmicInverseRetraction(),
    )
end

"""
    TangentSpaceTransformer(p = :mean)

Unsupervised model for transforming data to tangent space.
`p` is either equal to `:mean`, in which case data will be transformed to the tangent
space at the mean of points given to `fit`, or a specific point on a manifold (represented as a tuple).
"""
mutable struct TangentSpaceTransformer <: Unsupervised
    p::Union{Symbol,MLJManifoldPoint}
    retraction::AbstractRetractionMethod
    inverse_retraction::AbstractInverseRetractionMethod
    basis::ManifoldsBase.AbstractBasis
    features::AbstractVector{Symbol} # if not empty only these features will be transformed
end

function TangentSpaceTransformer()
    return TangentSpaceTransformer(
        :mean,
        ExponentialRetraction(),
        LogarithmicInverseRetraction(),
        DefaultOrthonormalBasis(),
        Symbol[],
    )
end

function univariate_to_tspace_fit(
    M::Manifold,
    p,
    v;
    retraction::AbstractRetractionMethod,
    inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
    basis = DefaultOrthonormalBasis(),
)
    if p === :mean
        point = (
            mean(
                M,
                map(q -> q[1], v);
                retraction = retraction,
                inverse_retraction = inverse_retraction,
            ),
            M,
        )
    else
        point = p
    end
    fitresult = (point..., basis)
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end

function univariate_to_tspace_transform(
    M::Manifold,
    p,
    basis,
    data,
    inverse_retraction::AbstractInverseRetractionMethod,
)
    inv_retrs = map(
        q -> get_coordinates(M, p, inverse_retract(M, p, q[1], inverse_retraction), basis),
        data,
    )
    return map(i -> map(u -> inv_retrs[u][i], 1:length(inv_retrs)), 1:manifold_dimension(M))
end

function univariate_to_tspace_inverse_transform(
    M::Manifold,
    p,
    basis,
    data,
    retraction::AbstractRetractionMethod,
)
    arr_data = Array(data)
    retrs = map(
        i -> retract(M, p, get_vector(M, p, arr_data[i, :], basis), retraction),
        axes(arr_data, 1),
    )
    return map(u -> (u, M), retrs)
end

function MLJBase.fit(transformer::TangentSpaceTransformer, verbosity::Int, X)
    is_univariate = !Tables.istable(X)

    if is_univariate
        M = X[1][2]
        return (
            (
                is_univariate = true,
                fitresult = univariate_to_tspace_fit(
                    M,
                    transformer.p,
                    X;
                    basis = transformer.basis,
                    retraction = transformer.retraction,
                    inverse_retraction = transformer.inverse_retraction,
                )[1],
            ),
            nothing,
            nothing,
        )
    end

    all_features = Tables.schema(X).names
    feature_scitypes = collect(elscitype(selectcols(X, c)) for c in all_features)
    if isempty(transformer.features)
        cols_to_fit = filter!(collect(eachindex(all_features))) do j
            return feature_scitypes[j] <: MLJScientificTypes.ManifoldPoint
        end
    else
        issubset(transformer.features, all_features) ||
            @warn "Some specified features not present in table to be fit. "
        cols_to_fit = filter!(collect(eachindex(all_features))) do j
            return (all_features[j] in transformer.features) &&
                   feature_scitypes[j] <: MLJScientificTypes.ManifoldPoint
        end
    end
    fitresult_given_feature = Dict{Symbol,Any}()

    isempty(cols_to_fit) && verbosity > -1 && @warn "No features to transform"
    for j in cols_to_fit
        col_data = selectcols(X, j)
        col_fitresult, cache, report = univariate_to_tspace_fit(
            col_data[1][2],
            transformer.p,
            col_data;
            retraction = transformer.retraction,
            inverse_retraction = transformer.inverse_retraction,
            basis = transformer.basis,
        )
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
function MLJBase.transform(
    transformer::UnivariateTangentSpaceTransformer,
    fitresult,
    p::MLJManifoldPoint,
)
    q, M, basis = fitresult
    X = inverse_retract(M, q, p[1], transformer.inverse_retraction)
    coeffs = get_coefficients(M, q, X, basis)
    new_features = [Symbol("X_$i") for i in 1:manifold_dimension(M)]
    named_cols = NamedTuple{tuple(new_features...)}(tuple(coeffs)...)
    return MLJBase.table(named_cols)
end

# for transforming vector:
function MLJBase.transform(transformer::TangentSpaceTransformer, fitresult, ps)
    is_univariate = fitresult.is_univariate

    if is_univariate
        M = ps[1][2]
        return univariate_to_tspace_transform(
            M,
            fitresult.fitresult[1],
            transformer.basis,
            ps,
            transformer.inverse_retraction,
        )
    end

    features_to_be_transformed = keys(fitresult.fitresult_given_feature)

    all_features = Tables.schema(ps).names

    all(e -> e in all_features, features_to_be_transformed) ||
        error("Attempting to transform data with incompatible feature labels.")

    new_features = Symbol[]
    new_cols = []


    for ftr in all_features
        ftr_data = selectcols(ps, ftr)
        if ftr in features_to_be_transformed
            fgf = fitresult.fitresult_given_feature[ftr]
            M = fgf[2]
            feature_names = [Symbol("$(ftr)_$i") for i in 1:manifold_dimension(M)]
            append!(new_features, feature_names)
            cols = univariate_to_tspace_transform(
                M,
                fgf[1],
                fgf[3],
                ftr_data,
                transformer.inverse_retraction,
            )
            append!(new_cols, cols)
        else
            push!(new_features, ftr)
            push!(new_cols, ftr_data)
        end
    end

    named_cols = NamedTuple{tuple(new_features...)}(tuple(new_cols...))

    return MLJBase.table(named_cols, prototype = ps)
end

function MLJBase.inverse_transform(transformer::TangentSpaceTransformer, fitresult, X)
    is_univariate = fitresult.is_univariate

    if is_univariate
        return inverse_transform(transformer, fitresult.fitresult, X)
    end

    features_transformed = keys(fitresult.fitresult_given_feature)
    features_processed = Symbol[]

    all_features = Tables.schema(X).names

    new_features = Symbol[]
    new_cols = []

    for ftr in all_features
        sftc = string(ftr)
        last_underscore = findlast('_', sftc)
        prefix = last_underscore === nothing ? ftr : Symbol(sftc[1:(last_underscore - 1)])
        if prefix in features_transformed
            if prefix in features_processed
                continue
            end
            fgf = fitresult.fitresult_given_feature[prefix]
            M = fgf[2]
            feature_names = [Symbol("$(prefix)_$i") for i in 1:manifold_dimension(M)]
            push!(new_features, prefix)
            ftr_data = selectcols(X, feature_names)
            new_col = univariate_to_tspace_inverse_transform(
                M,
                fgf[1],
                fgf[3],
                ftr_data,
                transformer.retraction,
            )
            push!(new_cols, new_col)
            push!(features_processed, prefix)
        else
            ftr_data = selectcols(X, ftr)
            push!(new_features, ftr)
            push!(new_cols, ftr_data)
        end
    end
    named_cols = NamedTuple{tuple(new_features...)}(tuple(new_cols...))

    return MLJBase.table(named_cols, prototype = X)
end


# for single values:
function MLJBase.inverse_transform(
    transformer::UnivariateTangentSpaceTransformer,
    fitresult,
    coeffs,
)
    q, M, basis = fitresult
    X = get_vector(M, q, coeffs, basis)
    p = retract(M, q, X, transformer.retraction)
    return (p, M)
end

# for vectors:
function MLJBase.inverse_transform(
    transformer::UnivariateTangentSpaceTransformer,
    fitresult,
    w::AbstractVector,
)
    return [MLJBase.inverse_transform(transformer, fitresult, y) for y in w]
end
