function baseparams(
    ::Type{GARegressor},
    fiteval,
    size_pop;
    select,
    crossover,
    p_mutate,
    init=:inverse,
    testonly=false,
)
    return Dict(
        :n_iter => ifelse(testonly, 10, 1000),
        :size_pop => size_pop,
        :fiteval => fiteval,
        # dgmodel
        :x_min => 0.0,
        :x_max => 1.0,
        :nmatch_min => 2,
        :n_iter_earlystop => 500,
        :init => ifelse(init == :random, :custom, :inverse),
        :init_sample_fname => "../2024-gecco-tasks/2024-01-09T16-54-46-439387-kdata/",
        # Ain't nobody got time for safety (this is still pretty safe, just
        # don't change any of the files in the `init_sample_fname` folder).
        :init_unsafe => true,
        # These seem to be sensible defaults.
        :init_length_min => 3,
        :init_length_max => 50,
        # These are ignored anyways since we use `init_sample_fname`.
        :init_spread_min => 0.1,
        :init_params_spread_a => 1.0,
        :init_params_spread_b => 1.0,
        # init_spread_max=Inf,
        # init_rate_coverage_min=0.9,
        # If we don't use crossover, we want more add/rm mutation.
        :mutate_p_add => p_mutate,
        :mutate_p_rm => p_mutate,
        :mutate_rate_mut => 1.0,
        :mutate_rate_std => 0.05,
        # If crossover is :off, set it to spatial but then set crossover
        # probability to 0.
        :recomb => ifelse(crossover == :off, :spatial, crossover),
        :recomb_rate => ifelse(crossover == :off, 0.0, 0.3),
        :select => select,
        # TODO Consider to optimize this
        :select_width_window => 7,
        # TODO Consider to select a somewhat informed value instead of
        # ryerkerk2020's
        :select_lambda_window => 0.004,
        :select_size_tournament => 4,
    )
end

function mkspace_mga(model, DX)
    space::Vector{ParamRange} = [
        range(model, :mutate_p_add; lower=0.01, upper=1.0),
        range(model, :mutate_p_rm; lower=0.01, upper=1.0),
        # range(
        #     model,
        #     :mutate_rate_mut;
        #     values=collect(0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0),
        # ),
        # range(model, :mutate_rate_std; lower=0.001, upper=0.2, scale=:log),
    ]
    # In basemodel, we set recomb_rate to 0.0 if crossover==:off.
    if model.recomb_rate != 0.0
        push!(space, range(model, :recomb_rate; lower=0.01, upper=1.0))
    end
    if model.select == :tournament
        push!(
            space,
            range(
                model,
                :select_size_tournament;
                values=collect(1:(div(model.size_pop, 4))),
            ),
        )
    end
    return space
end

function blacklist(::Type{GARegressor})
    return [:rng, :dgmodel]
end

function fixparams!(::Type{GARegressor}, params)
    params[:fiteval] = Symbol(params[:fiteval])
    params[:init] = Symbol(params[:init])
    params[:recomb] = Symbol(params[:recomb])
    params[:select] = Symbol(params[:select])
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
    postfix,
    size_pop;
    select,
    crossover,
    p_mutate,
    init=:inverse,
    testonly=false,
)
    return Variant(;
        label="MGA$size_pop-$postfix",
        label_family="GARegressor",
        type_model=GARegressor,
        params=baseparams(
            GARegressor,
            :NegAIC,
            size_pop;
            select=select,
            crossover=crossover,
            p_mutate=p_mutate,
            init=init,
            testonly=testonly,
        ),
        mkspace=nothing,
        additional=[],
    )
end
