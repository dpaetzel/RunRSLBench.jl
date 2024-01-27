function basemodel(
    ::Type{GARegressor},
    fiteval,
    size_pop,
    crossover;
    testonly=false,
)
    return GARegressor(;
        # TODO Early stopping would be a nice feature
        n_iter=ifelse(testonly, 10, 1000),
        size_pop=size_pop,
        fiteval=fiteval,
        # TODO dgmodel
        x_min=0.0,
        x_max=1.0,
        nmatch_min=2,
        n_iter_earlystop=100,
        init=:inverse,
        init_sample_fname="../2024-gecco-tasks/2024-01-09T16-54-46-439387-kdata/",
        # Ain't nobody got time for safety (this is still pretty safe, just
        # don't change any of the files in the `init_sample_fname` folder).
        init_unsafe=true,
        # These seem to be sensible defaults.
        init_length_min=3,
        init_length_max=50,
        # These are ignored anyways since we use `init_sample_fname`.
        # init_spread_min=0.1,
        # init_params_spread_a=1.0,
        # init_params_spread_b=1.0,
        # init_spread_max=Inf,
        # init_rate_coverage_min=0.9,
        # If we don't use crossover, we want more add/rm mutation.
        # Optimized.
        # mutate_p_add=ifelse(crossover, 0.05, 0.4),
        # Optimized.
        # mutate_p_rm=ifelse(crossover, 0.05, 0.4),
        mutate_rate_mut=1.0,
        mutate_rate_std=0.05,
        # Optimized.
        recomb_rate=ifelse(crossover, 0.8, 0.0),
        # Optimized.
        # select_width_window=7,
        # TODO Consider to select a somewhat informed value instead of
        # ryerkerk2020's
        select_lambda_window=0.004,
    )
end

function mkspace_mga(model, DX)
    space = [
        range(model, :select_width_window; values=collect(7:2:11)),
        range(model, :mutate_p_add; lower=0.01, upper=1.0),
        range(model, :mutate_p_rm; lower=0.01, upper=1.0),
        # range(
        #     model,
        #     :mutate_rate_mut;
        #     values=collect(0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0),
        # ),
        # range(model, :mutate_rate_std; lower=0.001, upper=0.2, scale=:log),
    ]
    if model.recomb_rate != 0.0
        push!(space, range(model, :recomb_rate; lower=0.01, upper=1.0))
    end
    return space
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
        "MGA$size_pop",
        "GARegressor",
        basemodel(
            GARegressor,
            :NegAIC,
            size_pop,
            crossover;
            testonly=testonly,
        ),
        mkspace_mga,
        [],
    )
end
