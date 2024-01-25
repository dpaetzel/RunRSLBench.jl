function basemodel(
    ::Type{GARegressor},
    fiteval,
    size_pop,
    crossover;
    testonly=false,
)
    return GARegressor(;
        # TODO Early stopping would be a nice feature
        n_iter=ifelse(testonly, 10, 200),
        size_pop=size_pop,
        fiteval=fiteval,
        # TODO dgmodel
        x_min=0.0,
        x_max=1.0,
        # Optimized.
        # nmatch_min=2,
        init=:inverse,
        init_sample_fname="../2024-gecco-tasks/2024-01-09T16-54-46-439387-kdata/",
        # Ain't nobody got time for safety (this is still pretty safe, just
        # don't change any of the files in the `init_sample_fname` folder).
        init_unsafe=true,
        # These seem sensible defaults.
        init_length_min=3,
        init_length_max=50,
        # These are ignored anyways since we use `init_sample_fname`.
        # init_spread_min=0.1,
        # init_params_spread_a=1.0,
        # init_params_spread_b=1.0,
        # init_spread_max=Inf,
        # init_rate_coverage_min=0.9,
        # If we don't use crossover, we want more add/rm mutation.
        mutate_p_add=ifelse(crossover, 0.05, 0.4),
        mutate_p_rm=ifelse(crossover, 0.05, 0.4),
        mutate_rate_mut=1.0,
        mutate_rate_std=0.05,
        recomb_rate=ifelse(crossover, 0.9, 0.0),
        # Optimized.
        # select_width_window=7,
        # TODO Consider to select a somewhat informed value instead of
        # ryerkerk2020's
        select_lambda_window=0.004,
    )
end

function mkspace_mga(model, DX)
    return [
        range(model, :select_width_window; values=collect(3:2:15)),
        range(model, :nmatch_min; values=[2, 3, 5, 8]),
    ]
end

function blacklist(::GARegressor)
    return [:rng, :dgmodel]
end

function fixparams!(::Type{GARegressor}, params)
    params[:fiteval] = Symbol(params[:fiteval])
    params[:init] = Symbol(params[:init])
    if params[:init_spread_max] == nothing
        params[:init_spread_max] = Inf
    end

    return params
end

function userextras(::GARegressor)
    function _userextras(model, fitted_params_per_fold)
        return MLJ.recursive_getproperty.(
            fitted_params_per_fold,
            Ref(:(fitresult.best.fitness)),
        )
    end

    return _userextras
end

function mkvariant(
    ::Type{GARegressor},
    size_pop,
    dgmodel;
    crossover=true,
    testonly=false,
)
    return Variant(
        "MGA$size_pop-posterior",
        "GARegressor",
        basemodel(
            GARegressor,
            :posterior,
            size_pop,
            crossover;
            testonly=testonly,
        ),
        mkspace_mga,
        [
            Override(
                "MGA$size_pop-similarity",
                [:fiteval => :similarity, :dgmodel => dgmodel],
            ),
        ],
    )
end
