function baseparams(::Type{DT})
    # Note things we comment out here are being optimized (see `mkmkspace_dt`).
    return Dict(
        # max_depth=-1,
        :min_samples_leaf => 5,
        # min_samples_split=2,
        :min_purity_increase => 0.0,
        # Always consider all features.
        :n_subfeatures => 0,
        :post_prune => false,
        :merge_purity_threshold => 1.0,
        :feature_importance => :impurity,
    )
end

"""
Given input space dimensionality and a minimum and maximum number of rules,
define the hyperparameter space for DT algorithms.

Note that this returns a function with parameters `pipe` and `DX` which then
returns a named tuple containing the tuning space and a blacklist of parameters
which should not be stored for later instantiation (i.e. RNG objects etc.).
"""
function mkmkspace_dt(K_min, K_max, N)
    max_depth_min = max(1, ceil(log2(K_min / 2)))
    max_depth_max = max(max_depth_min + 1, ceil(log2(K_max / 2)))

    function mkspace_dt(pipe, DX)
        return [
            range(pipe, :max_depth; lower=max_depth_min, upper=max_depth_max),
            range(
                pipe,
                :min_samples_split;
                lower=ceil(0.001 * N),
                upper=ceil(0.05 * N),
                # scale = :log,
            ),
        ]
    end

    return mkspace_dt
end

function blacklist(::Type{DT})
    return [:rng]
end

function fixparams!(::Type{DT}, params)
    return params[:feature_importance] = Symbol(params[:feature_importance])
end

function userextras(::DT)
    function _userextras(model, fitted_params_per_fold)
        # TODO Probably unnecessary
    end

    return _userextras
end

function mkvariant(::Type{DT}, n, nrules_min, nrules_max; testonly=false)
    return Variant(;
        label="DT$nrules_min-$nrules_max",
        label_family="DT",
        type_model=DT,
        params=baseparams(DT),
        mkspace=mkmkspace_dt(nrules_min, nrules_max, n),
        additional=[],
    )
end
