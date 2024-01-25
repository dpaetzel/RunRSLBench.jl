function basemodel(::Type{XCSFRegressor}, n, pop_size; testonly=false)
    @warn "Will have to base theta_sub on number of training data " *
          "point/expected number of rules"
    # Note things we comment out here are being optimized (see `mkspace_xcsf`).
    return XCSFRegressor(;
        # Whether to seed the population with random rules.
        pop_init=true,
        # Number of trials to execute for each xcs.fit(). We just take a large
        # number and hope that XCSF converges.
        max_trials=ifelse(testonly, 10, 200000),
        # Maximum population size.
        # Note that we do not hyperparameter optimize `pop_size` because that
        # would be unfair. We will instead choose 4 fixed values.
        pop_size=pop_size,
        # Whether to perform action set subsumption.
        set_subsumption=true,
        # Minimum experience of a rule to become a subsumer.
        theta_sub=100,
        loss_func="mae",
        # Rule error below which accuracy is set to 1.
        # e0=0.01,
        # Accuracy offset for rules with error above e0 (1=disabled).
        alpha=0.1,
        # Accuracy slope for rules with error above e0.
        # nu=5,
        # Learning rate for updating error, fitness, and set size.
        # beta=0.1,
        # Rate of fitness-worst rules with an increased deletion probability.
        delta=0.1,
        # Minimum experience before fitness used in probability of deletion.
        theta_del=20,
        # Initial rule fitness.
        init_fitness=0.01,
        # Initial rule error.
        init_error=0.0,
        # Parental selection.
        ea_select_type="tournament",
        # Fraction of set size for tournament parental selection.
        ea_select_size=0.4,
        # Average set time between EA invocations.
        ea_theta_ea=50,
        # Number of offspring to create each EA invocation (use multiples of 2).
        ea_lambda=2,
        # Probability of applying crossover.
        ea_p_crossover=0.8,
        # Rate to reduce offspring error by (1=disabled).
        ea_err_reduc=1.0,
        # Rate to reduce offspring fitness by (1=disabled).
        ea_fit_reduc=0.1,
        # Whether to try and subsume offspring rules.
        ea_subsumption=true,
        # Whether to reset offspring predictions instead of copying.
        ea_pred_reset=false,
        # Minimum spread of conditions randomly generated during population
        # initialization or covering.
        # condition_spread_min=0.2,
        # Minimum value of each input space dimension (use min-max
        # normalization!) including some wiggle room for the GA.
        x_min=-0.1,
        # Maximum value of each input space dimension (use min-max
        # normalization!) including some wiggle room for the GA.
        x_max=1.1,
    )
end

function mkspace_xcsf(model, DX)
    # I personally find it very annoying that I have to provide `model` to this.
    return (;
        space=[
            range(model, :e0; lower=0.01, upper=0.2, scale=:log),
            # TODO Consider even higher beta values as well?
            range(model, :beta; lower=0.001, upper=0.3, scale=:log),
            range(model, :nu; lower=1, upper=10),
            # Note that the actual maximum spread possible is 0.6 since XCSF's
            # conditions are allowed to be in [-0.1, 1.1] to allow some wiggle
            # room for the GA (data is in [0.0, 1.0] only).
            range(model, :condition_spread_min; lower=0.01, upper=0.5),
            # range(model, :ea_select_size; lower=0.1, upper=0.8),
            # range(model, :ea_p_crossover; lower=0.0, upper=1.0),
        ],
        blacklist=[:rng],
    )
end

function fixparams!(::Type{XCSFRegressor}, params)
    return params
end

function userextras(::XCSFRegressor)
    function _userextras(model, fitted_params_per_fold)
        # TODO Probably unnecessary
    end

    return _userextras
end

function mkvariant(::Type{XCSFRegressor}, n, pop_size; testonly=false)
    return Variant(
        "XCSF$pop_size",
        "XCSFRegressor",
        basemodel(XCSFRegressor, n, pop_size; testonly=testonly),
        mkspace_xcsf,
        [],
    )
end
