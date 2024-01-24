function basemodel(
    ::Type{GARegressor},
    fiteval,
    size_pop,
    crossover;
    testonly=false,
)
    return GARegressor(;
        n_iter=ifelse(testonly, 10, 100),
        size_pop=size_pop,
        fiteval=fiteval,
        # TODO dgmodel
        x_min=0.0,
        x_max=1.0,
        nmatch_min=2,
        init=:inverse,
        init_sample_fname="../2024-gecco-tasks/2024-01-09T16-54-46-439387-kdata/",
        # Ain't nobody got time for safety (this is still pretty safe, just
        # don't change any of the files in the `init_sample_fname` folder).
        init_unsafe=true,
        init_length_min=3,
        init_length_max=50,
        # These are ignored anyways since we use `init_sample_fname`.
        # init_spread_min=0.1,
        # init_params_spread_a=1.0,
        # init_params_spread_b=1.0,
        # init_spread_max=Inf,
        # init_rate_coverage_min=0.9,
        # TODO Check with n_iter for being sensible
        mutate_p_add=ifelse(crossover, 0.05, 0.4),
        mutate_p_rm=ifelse(crossover, 0.05, 0.4),
        mutate_rate_mut=1.0,
        mutate_rate_std=0.05,
        recomb_rate=ifelse(crossover, 0.9, 0.0),
        # TODO Select sensible value (probably interacts with init_length_min/max)
        select_width_window=7,
        # TODO Select sensible value
        select_lambda_window=0.004,
    )
end

function mkspace_mga(model, DX)
    return (;
        space=[range(model, :recomb_rate; lower=0.4, upper=1.0)],
        blacklist=[:rng],
    )
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
    fiteval,
    size_pop;
    crossover=true,
    testonly=false,
)
    return (;
        label="MGA$size_pop-$fiteval",
        family="GARegressor",
        model=basemodel(
            GARegressor,
            fiteval,
            size_pop,
            crossover;
            testonly=testonly,
        ),
        mkspace=mkspace_mga,
    )
end