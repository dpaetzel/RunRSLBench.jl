#! format: off
module Transformers

# TODO Cleanup this code
# TODO Add `ignore` and other features that `Standardizer` has

using MLJ
# Required for LittleDict prior to high enough MLJ versions (which are blocked
# in 2023-11 by MLJTuning and TreeParzen).
using OrderedCollections
using Tables

export UnivariateMinMaxScaler, MinMaxScaler

# Some of the credit goes to
#
# - https://github.com/alan-turing-institute/MLJ.jl/issues/816
# - https://github.com/JuliaAI/MLJModels.jl/blob/dev/src/builtins/Transformers.jl#L491
#
# Actually, large parts are just the `MLJModels.builtins.Transformers.jl` code
# but adapted and (for now) simplified.
mutable struct UnivariateMinMaxScaler <: Unsupervised
    # TODO Add support for intervals other than [0, 1]
end

function MLJ.fit(
    transformer::UnivariateMinMaxScaler,
    verbosity::Int,
    v::AbstractVector{T},
) where {T<:Real}
    fitresult = (minimum(v), maximum(v))
    cache = nothing
    report = NamedTuple()
    return (fitresult, cache, report)
end

function MLJ.fitted_params(::UnivariateMinMaxScaler, fitresult)
    return (xmin = fitresult[1], xmax = fitresult[2])
end

# Transform a single value.
function MLJ.transform(transformer::UnivariateMinMaxScaler, fitresult, x::Real)
    xmin, xmax = fitresult
    x_transformed = (x .- xmin) ./ (xmax .- xmin)
    return x_transformed
end

# Transform a vector.
function MLJ.transform(transformer::UnivariateMinMaxScaler, fitresult, v::AbstractVector)
    return [MLJ.transform(transformer, fitresult, x) for x in v]
end

# Inverse transform a single value.
function MLJ.inverse_transform(
    transformer::UnivariateMinMaxScaler,
    fitresult,
    x_transformed::Real,
)
    xmin, xmax = fitresult
    # x_transformed = (x .- xmin) ./ (xmax .- xmin)
    # x_transformed .* (xmax .- xmin) = x .- xmin
    x = x_transformed .* (xmax .- xmin) .+ xmin
    return x
end

# Inverse transform a vector.
function MLJ.inverse_transform(
    transformer::UnivariateMinMaxScaler,
    fitresult,
    v_transformed::AbstractVector,
)
    return [
        MLJ.inverse_transform(transformer, fitresult, x_transformed) for
        x_transformed in v_transformed
    ]
end

# The remainder of this file is just copied and names adjusted from the
# definition of Standardizer at
# https://github.com/JuliaAI/MLJModels.jl/blob/dev/src/builtins/Transformers.jl#L491 .

mutable struct MinMaxScaler <: Unsupervised
    # features to be standardized; empty means all
    features::Union{AbstractVector{Symbol},Function}
    ignore::Bool # features to be ignored
    ordered_factor::Bool
    count::Bool
end

# keyword constructor
"""
    MinMaxScaler(; <keyword arguments>)

# Arguments
- `features`::Union{AbstractVector{Symbol},Function}: If empty symbol (the
  default), scale all features. *Should* behave the same as the `features`
  keyword of `Standardizer`.
"""
function MinMaxScaler(;
    features::Union{AbstractVector{Symbol},Function} = Symbol[],
    ignore::Bool = false,
    ordered_factor::Bool = false,
    count::Bool = false,
)
    transformer = MinMaxScaler(features, ignore, ordered_factor, count)
    message = MLJ.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MLJ.clean!(transformer::MinMaxScaler)
    err = ""
    if (
        typeof(transformer.features) <: AbstractVector{Symbol} &&
        isempty(transformer.features) &&
        transformer.ignore
    )
        err *= "Features to be ignored must be specified in features field."
    end
    return err
end

function MLJ.fit(transformer::MinMaxScaler, verbosity::Int, X)

    # if not a table, it must be an abstract vector, eltpye AbstractFloat:
    is_univariate = !Tables.istable(X)

    # are we attempting to standardize Count or OrderedFactor?
    is_invertible = !transformer.count && !transformer.ordered_factor

    # initialize fitresult:
    fitresult_given_feature = LittleDict{Symbol,Tuple{Float64,Float64}}()

    # special univariate case:
    if is_univariate
        fitresult_given_feature[:unnamed] =
            MLJ.fit(UnivariateMinMaxScaler(), verbosity - 1, X)[1]
        return (
            is_univariate = true,
            is_invertible = true,
            fitresult_given_feature = fitresult_given_feature,
        ),
        nothing,
        nothing
    end

    all_features = Tables.schema(X).names
    feature_scitypes = collect(elscitype(selectcols(X, c)) for c in all_features)
    scitypes = Vector{Type}([Continuous])
    transformer.ordered_factor && push!(scitypes, OrderedFactor)
    transformer.count && push!(scitypes, Count)
    AllowedScitype = Union{scitypes...}

    # determine indices of all_features to be transformed
    if transformer.features isa AbstractVector{Symbol}
        if isempty(transformer.features)
            cols_to_fit = filter!(eachindex(all_features) |> collect) do j
                return feature_scitypes[j] <: AllowedScitype
            end
        else
            !issubset(transformer.features, all_features) &&
                verbosity > -1 &&
                @warn "Some specified features not present in table to be fit. "
            cols_to_fit = filter!(eachindex(all_features) |> collect) do j
                return ifelse(
                    transformer.ignore,
                    !(all_features[j] in transformer.features) &&
                    feature_scitypes[j] <: AllowedScitype,
                    (all_features[j] in transformer.features) &&
                    feature_scitypes[j] <: AllowedScitype,
                )
            end
        end
    else
        cols_to_fit = filter!(eachindex(all_features) |> collect) do j
            return ifelse(
                transformer.ignore,
                !(transformer.features(all_features[j])) &&
                feature_scitypes[j] <: AllowedScitype,
                (transformer.features(all_features[j])) &&
                feature_scitypes[j] <: AllowedScitype,
            )
        end
    end
    fitresult_given_feature = Dict{Symbol,Tuple{Float64,Float64}}()

    isempty(cols_to_fit) && verbosity > -1 && @warn "No features to standarize."

    # fit each feature and add result to above dict
    verbosity > 1 && @info "Features standarized: "
    for j in cols_to_fit
        col_data = if (feature_scitypes[j] <: OrderedFactor)
            coerce(selectcols(X, j), Continuous)
        else
            selectcols(X, j)
        end
        col_fitresult, _, _ = MLJ.fit(UnivariateMinMaxScaler(), verbosity - 1, col_data)
        fitresult_given_feature[all_features[j]] = col_fitresult
        verbosity > 1 && @info "  :$(all_features[j])    xmin=$(col_fitresult[1])  " *
              "xmax=$(col_fitresult[2])"
    end

    fitresult = (
        is_univariate = false,
        is_invertible = is_invertible,
        fitresult_given_feature = fitresult_given_feature,
    )
    cache = nothing
    report = (features_fit = keys(fitresult_given_feature),)

    return fitresult, cache, report
end

function MLJ.fitted_params(::MinMaxScaler, fitresult)
    is_univariate, _, dic = fitresult
    is_univariate && return fitted_params(UnivariateMinMaxScaler(), dic[:unnamed])
    features_fit = keys(dic) |> collect
    zipped = map(ftr -> dic[ftr], features_fit)
    means, stds = zip(zipped...) |> collect
    return (; features_fit, means, stds)
end

function MLJ.transform(::MinMaxScaler, fitresult, X)
    return _scale(transform, fitresult, X)
end

function MLJ.inverse_transform(::MinMaxScaler, fitresult, X)
    fitresult.is_invertible || error(
        "Inverse standardization is not supported when `count=true` " *
        "or `ordered_factor=true` during fit. ",
    )
    return _scale(inverse_transform, fitresult, X)
end

function _scale(operation, fitresult, X)
    # `fitresult` is dict of column fitresults, keyed on feature names
    is_univariate, _, fitresult_given_feature = fitresult

    if is_univariate
        univariate_fitresult = fitresult_given_feature[:unnamed]
        return operation(UnivariateMinMaxScaler(), univariate_fitresult, X)
    end

    features_to_be_transformed = keys(fitresult_given_feature)

    all_features = Tables.schema(X).names

    all(e -> e in all_features, features_to_be_transformed) ||
        error("Attempting to transform data with incompatible feature labels.")

    col_transformer = UnivariateMinMaxScaler()

    cols = map(all_features) do ftr
        ftr_data = selectcols(X, ftr)
        if ftr in features_to_be_transformed
            col_to_transform = coerce(ftr_data, Continuous)
            operation(col_transformer, fitresult_given_feature[ftr], col_to_transform)
        else
            ftr_data
        end
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))

    return MLJ.table(named_cols; prototype = X)
end

end
