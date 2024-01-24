module XCSF

using MLJModelInterface: MLJModelInterface
const MMI = MLJModelInterface
using PyCall
using Random
using Tables

export XCSFRegressor

const PYXCSF = PyNULL()
const PYEarlyStoppingCallback = PyNULL()

function __init__()
    copy!(PYXCSF, pyimport("xcsf")["XCS"])
    copy!(PYEarlyStoppingCallback, pyimport("xcsf")["EarlyStoppingCallback"])
    return nothing
end

# Defines the XCSF struct, a clean! method and a keyword constructor.
MMI.@mlj_model mutable struct XCSFRegressor <: MMI.Deterministic
    # Whether to seed the population with random rules.
    pop_init::Bool = true
    # Number of trials to execute for each xcs.fit().
    max_trials::Int32 = 1000::(_ > 0)
    # Maximum population size.
    pop_size::Int32 = 500::(_ > 0)
    # Whether to perform action set subsumption.
    set_subsumption::Bool = true
    # Minimum experience of a rule to become a subsumer.
    theta_sub::Int32 = 100::(_ >= 0)
    loss_func::String = "mae"::(_ ∈ ["mae", "mse", "rmse"])
    # Rule error below which accuracy is set to 1.
    e0::Float64 = 0.01::(_ >= 0)
    # Accuracy offset for rules with error above e0 (1=disabled).
    alpha::Float64 = 0.1::(1 >= _ >= 0)
    # Accuracy slope for rules with error above e0.
    nu::Int32 = 5::(_ > 0)
    # Learning rate for updating error, fitness, and set size.
    beta::Float64 = 0.1::(1 >= _ > 0)
    # Rate of fitness-worst rules with an increased deletion probability.
    delta::Float64 = 0.1::(1 >= _ >= 0)
    # Minimum experience before fitness used in probability of deletion.
    theta_del::Int32 = 20::(_ >= 0)
    # Initial rule fitness.
    init_fitness::Float64 = 0.01::(_ >= 0)
    # Initial rule error.
    init_error::Float64 = 0.0::(_ >= 0)
    # Parental selection.
    ea_select_type::String = "tournament"::(_ ∈ ["roulette", "tournament"])
    # Fraction of set size for tournament parental selection.
    ea_select_size::Float64 = 0.4::(_ > 0)
    # Average set time between EA invocations.
    ea_theta_ea::Int32 = 50::(_ > 0)
    # Number of offspring to create each EA invocation (use multiples of 2).
    ea_lambda::Int32 = 2::(_ > 0)
    # Probability of applying crossover.
    ea_p_crossover::Float64 = 0.8::(1 >= _ >= 0)
    # Rate to reduce offspring error by (1=disabled).
    ea_err_reduc::Float64 = 1.0::(1 >= _ >= 0)
    # Rate to reduce offspring fitness by (1=disabled).
    ea_fit_reduc::Float64 = 0.1::(1 >= _ >= 0)
    # Whether to try and subsume offspring rules.
    ea_subsumption::Bool = false
    # Whether to reset offspring predictions instead of copying.
    ea_pred_reset::Bool = false
    # Minimum spread of conditions randomly generated during population
    # initialization or covering.
    condition_spread_min::Float64 = 0.1::(_ >= 0)
    # Minimum value of each input space dimension (use min-max normalization!).
    x_min::Float64 = -0.1
    # Maximum value of each input space dimension (use min-max normalization!).
    # TODO Enforce x_max > x_min
    x_max::Float64 = 1.1
    # RNG.
    rng::Union{AbstractRNG,Integer} = Random.GLOBAL_RNG
end

# https://github.com/JuliaAI/DecisionTree.jl/blob/8d565b562887f081c5076496b610dbea4c56dca9/src/DecisionTree.jl#L158
mkrng(rng::Random.AbstractRNG) = rng
mkrng(seed::T) where {T<:Integer} = Random.Xoshiro(seed)

function MMI.fit(model::XCSFRegressor, verbosity, X, y)
    Xmat = Tables.matrix(X)
    N, DX = size(Xmat)
    n_trials_per_epoch = Int(ceil(model.max_trials / 20))
    # TODO Add remaining relevant parameters here
    model_ = PYXCSF(;
        x_dim=DX,
        y_dim=1,
        # 1 for supervised learning.
        n_actions=1,
        # We parallelize on another level.
        omp_num_threads=1,
        # Since I'm not sure right now how RNGs interact from Julia to Python/C,
        # we'll simply use a seed here. This is the range of allowed NumPy
        # legacy RNG seeds.
        random_state=rand(mkrng(model.rng), 0:(2^32 - 1)),
        pop_init=model.pop_init,
        max_trials=model.max_trials,
        # We assess performance 20 times during training (relevant for Early
        # Stopping). This is a rather arbitrary number.
        perf_trials=n_trials_per_epoch,
        pop_size=model.pop_size,
        set_subsumption=model.set_subsumption,
        theta_sub=model.theta_sub,
        loss_func=model.loss_func,
        e0=model.e0,
        alpha=model.alpha,
        nu=model.nu,
        beta=model.beta,
        delta=model.delta,
        theta_del=model.theta_del,
        init_fitness=model.init_fitness,
        init_error=model.init_error,
        # Number of trials since its creation that a rule must match at least 1
        # input or be deleted. We disable this feature.
        m_probation=model.max_trials + 1000,
        # Rules should retain state across trials.
        stateful=true,
        # If enabled and overall system error is below e0, the largest of 2
        # roulette spins is deleted.
        compaction=false,
        ea=Dict(
            "select_type" => model.ea_select_type,
            "select_size" => model.ea_select_size,
            "theta_ea" => model.ea_theta_ea,
            "lambda" => model.ea_lambda,
            "p_crossover" => model.ea_p_crossover,
            "err_reduc" => model.ea_err_reduc,
            "fit_reduc" => model.ea_fit_reduc,
            "subsumption" => model.ea_subsumption,
            "pred_reset" => model.ea_pred_reset,
        ),
        # Supervised learning requires `"integer"` here.
        action=Dict("type" => "integer"),
        condition=Dict(
            "type" => "hyperrectangle_csr",
            "args" => Dict(
                "eta" => 0.0,
                "min" => model.x_min,
                "max" => model.x_max,
                "spread_min" => model.condition_spread_min,
            ),
        ),
        # TODO Support more than just constant models
        prediction=Dict("type" => "constant"),
    )
    model_.fit(
        Xmat,
        y;
        verbose=false,
        callbacks=[
            PYEarlyStoppingCallback(;
                # Which metric to monitor: {"train", "val"}. We simply monitor
                # train because we do not train-val split for now.
                monitor="train",
                # Trials with no improvement after which training will be stopped.
                # Note that early stopping is checked only after each epoch
                # anyways so we choose a number of epochs here.
                patience=2 * n_trials_per_epoch,
                # Whether to restore the best population after terminating. For
                # now I'll assume that compaction happens in the end and we
                # don't want do undo sensible stuff like that.
                # TODO Evaluate whether `restore_best` should be `true`.
                restore_best=true,
                # Minimum change to qualify as an improvement.
                min_delta=0,
                # Trials to wait before starting to monitor improvement.
                start_from=0,
                # Whether to display when an action is taken.
                # TODO Disable this, for debugging only
                verbose=verbosity >= 10,
            ),
        ],
    )

    fitresult = model_
    cache = nothing
    report = (; metrics=model_.get_metrics())

    return fitresult, cache, report
end

function MMI.predict(model::XCSFRegressor, fitresult, Xnew)
    Xnewmat = Tables.matrix(Xnew)
    model_ = fitresult
    yhat = model_.predict(Xnewmat)
    return vec(yhat)
end

MMI.input_scitype(::Type{<:XCSFRegressor}) = MMI.Table(MMI.Continuous)
MMI.target_scitype(::Type{<:XCSFRegressor}) = AbstractVector{MMI.Continuous}

# TODO If fully package, fill out remaining fields
# MMI.load_path(::Type{<:XCSFRegressor}) = ""
# MMI.package_name(::Type{<:XCSFRegressor}) = "Unknown"
# MMI.package_uuid(::Type{<:XCSFRegressor}) = "Unknown"
# MMI.package_url(::Type{<:XCSFRegressor}) = "unknown"
# MMI.is_pure_julia(::Type{<:XCSFRegressor}) = false
# MMI.package_license(::Type{<:XCSFRegressor}) = "unknown"

# TODO Maybe return user-friendly form of fitted parameters
# MMI.fitted_params(model::XCSFRegressor, fitresult) = fitresult

# Optional, to avoid redundant calculations when re-fitting machines associated with a model:
# MMI.update(model::XCSFRegressor, verbosity, old_fitresult, old_cache, X, y) =
#     MMI.fit(model, verbosity, X, y)

# Optional, to specify default hyperparameter ranges (for use in tuning).
# MMI.hyperparameter_ranges(XCSFRegressor) = (max_trials = (1, typemax(Int)))

# Optionally, to customized support for serialization of machines (see
# Serialization).
# MMI.save(filename, model::SomeModel, fitresult; kwargs...) = fitresult
# MMI.restore(filename, model::SomeModel, serializable_fitresult) -> serializable_fitresult

end
