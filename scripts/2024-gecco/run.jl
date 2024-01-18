using Comonicon
using Base64
using Dates
using DataFrames
using JSON
using TOML
using Missings
using MLFlowClient
using MLJ
using MLJTuning
using NPZ
using RSLModels.Intervals
using RSLModels.MLFlowUtils
using RSLModels.Transformers
using RSLModels.GARegressors
using RunRSLBench.RuleSupport
using RunRSLBench.XCSF
using Serialization
using Tables
# using TreeParzen
DT = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0

"""
Get MLflow client using authentication.
"""
function getmlf()
    config = open("config.toml", "r") do file
        return TOML.parse(read(file, String))
    end

    mlflow_url = "http://localhost:5000"

    username = config["database"]["user"]
    password = config["database"]["password"]
    encoded_credentials = base64encode("$username:$password")
    headers = Dict("Authorization" => "Basic $encoded_credentials")
    return MLFlow(mlflow_url; headers=headers)
end

function mkspace_xcsf(pipe, DX)
    return (;
        space=[
            range(pipe, :e0; lower=0.01, upper=0.2, scale=:log),
            # TODO Consider even higher beta values as well?
            range(pipe, :beta; lower=0.001, upper=0.3, scale=:log),
            # TODO Consider to use cube size
            range(pipe, :condition_spread_min; lower=0.01, upper=0.5),
            range(pipe, :ea_select_size; lower=0.1, upper=0.8),
            range(pipe, :ea_p_crossover; lower=0.0, upper=1.0),
        ],
        blacklist=[],
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
        return (;
            space=[
                range(
                    pipe,
                    :max_depth;
                    lower=max_depth_min,
                    upper=max_depth_max,
                ),
                range(
                    pipe,
                    :min_samples_split;
                    lower=ceil(0.001 * N),
                    upper=ceil(0.05 * N),
                    # scale = :log,
                ),
            ],
            blacklist=[:rng],
        )
    end

    return mkspace_dt
end

function mkspace_mga(model, DX)
    return (;
        space=[range(model, :recomb_rate; lower=0.4, upper=1.0)],
        blacklist=[:rng],
    )
end

function fixparams!(::Type{DT}, params)
    return params[:feature_importance] = Symbol(params[:feature_importance])
end

function fixparams!(::Type{XCSFRegressor}, params)
    return params
end

function fixparams!(::Type{GARegressor}, params)
    params[:fiteval] = Symbol(params[:fiteval])
    params[:init] = Symbol(params[:init])
    if params[:init_spread_max] == nothing
        params[:init_spread_max] = Inf
    end

    return params
end

function listvariants(N; testonly=false)
    xcsf = XCSFRegressor(;
        # TODO Set higher value
        max_trials=ifelse(testonly, 10, 200000),
        # Note that we do not hyperparameter optimize `pop_size` because that
        # would be unfair. We will instead choose 4 fixed values.
        pop_size=500,
    )

    dt = DT(;
        min_samples_leaf=1,
        min_purity_increase=0,
        # Always consider all features.
        n_subfeatures=0,
    )

    mga_mae = GARegressor(;
        n_iter=ifelse(testonly, 10, 100),
        size_pop=32,
        fiteval=:mae,
        x_min=0.0,
        x_max=1.0,
        nmatch_min=2,
        init=:inverse,
        # TODO Derive from DX
        init_length_min=3,
        # TODO Derive from DX
        init_length_max=30,
        # TODO Derive from DX
        init_spread_min=0.1,
        init_spread_max=Inf,
        init_params_spread_a=1.0,
        init_params_spread_b=1.0,
        # TODO Check how this is actually used (mutation, init, init_sample_fname, …)
        init_rate_coverage_min=0.9,
        # TODO Check with n_iter for being sensible
        mutate_p_add=0.05,
        mutate_p_rm=0.05,
        mutate_rate_mut=1.0,
        mutate_rate_std=0.05,
        recomb_rate=0.9,
        # TODO Select sensible value (probably interacts with init_length_min/max)
        select_width_window=7,
        # TODO Select sensible value
        select_lambda_window=0.004,
    )

    return [
        # ("XCSF100", "XCSFRegressor", xcsf, mkspace_xcsf),
        # TODO Deduplicate DT and dt (instead of dt provide `params_fixed` or similar)
        # ("DT2-50", "DT", dt, mkmkspace_dt(2, 50, N)),
        # ("DT2-100", "DT", dt, mkmkspace_dt(2, 100, N)),
        ("MGA-MAE", "GARegressor", mga_mae, mkspace_mga),
    ]
end

function userextras(::DT)
    function _userextras(model, fitted_params_per_fold)
        # TODO Probably unnecessary
    end

    return _userextras
end

function userextras(::XCSFRegressor)
    function _userextras(model, fitted_params_per_fold)
        # TODO Probably unnecessary
    end

    return _userextras
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

struct UserextraSelection <: MLJTuning.SelectionHeuristic end

function MLJTuning.best(::UserextraSelection, history)
    # We score using mean of CV here. I'm not that confident that this is the
    # best choice (e.g. median would be another option).
    scores = mean.(getproperty.(history, :userextras))
    index_best = argmin(scores)
    return history[index_best]
end

# Let's pirate some types. Julia, please forgive me.
MLJTuning.supports_heuristic(::LatinHypercube, ::UserextraSelection) = true

function tune(model, mkspace, X, y; testonly=false)
    N, DX = nrow(X), ncol(X)

    # TODO Refactor model being provided here after pipe
    sb = mkspace(model, DX)
    space = sb.space
    blacklist = sb.blacklist

    pipe = Pipeline(;
        scaler=MinMaxScaler(),
        model=TunedModel(;
            model=model,
            resampling=CV(; nfolds=5),
            # TODO Right now TreeParzen does not support nested hyperparameters (see my
            # issue on that).
            # tuning = MLJTreeParzenTuning(),
            tuning=LatinHypercube(; gens=30),
            range=space,
            measure=mae,
            selection_heuristic=ifelse(
                model isa GARegressor,
                UserextraSelection(),
                NaiveSelection(),
            ),
            # Number of models to evaluate. Note that without this, LatinHypercube fails
            # with an obscure error message.
            # TODO Increase number of models/make more fair wrt runtime
            n=ifelse(testonly, 2, 100),
            userextras=ifelse(
                model isa GARegressor,
                userextras(model),
                (_, _) -> nothing,
            ),
        ),
    )

    mach_tuned = machine(pipe, X, y)
    MLJ.fit!(mach_tuned; verbosity=1000)

    return mach_tuned, blacklist
end

function readdata(fname)
    @info "Reading train and test data from $fname …"
    data = npzread(fname)

    X, y = data["X"], data["y"]
    X = MLJ.table(X)

    X_test, y_test = data["X_test"], data["y_test"]
    X_test = MLJ.table(X_test)

    hash_task = data["hash"]

    @info "Read training data for task $(string(hash_task; base=16)) from $fname."
    return (X, y, hash_task, X_test, y_test)
end

"""
Optimize hyperparameters of each configured ML algorithms for the given learning
tasks, logging results to mlflow.

# Args

- `fnames`: Name of NPZ files that each containing training data for one
  learning task.

# Options

- `--name-run`: Give the runs performed (one per ML algorithm configured) the
  same specific name.

# Flags

- `--testonly`: Whether to perform very short runs that do not attempt to be
  competitive (useful for testing the pipeline).
"""
@cast function optparams(
    fnames...;
    testonly::Bool=false,
    name_run::Union{Missing,String}=missing,
)
    # TODO Random seeding
    # TODO Random seeding for XCSF

    for fname in fnames
        @info "Starting hyperparameter optimization for learning task $fname."
        X, y, hash_task, _, _ = readdata(fname)
        N, DX = nrow(X), ncol(X)

        mlf = getmlf()
        @info "Logging to mlflow tracking URI $(mlf.baseuri)."

        name_exp = "optparams"
        @info "Setting experiment name to $name_exp …"
        mlfexp = getorcreateexperiment(mlf, name_exp)

        for (label, family, model, mkspace) in
            listvariants(N; testonly=testonly)
            @info "Starting run …"
            mlfrun = createrun(mlf, mlfexp; run_name=name_run)
            name_run_final = mlfrun.info.run_name
            @info "Started run $name_run_final with id $(mlfrun.info.run_id)."

            logparam(
                mlf,
                mlfrun,
                Dict(
                    "algorithm.family" => string(family),
                    "algorithm.name" => label,
                    "task.hash" => hash_task,
                    "task.DX" => DX,
                    "task.N" => "N",
                ),
            )

            @info "Tuning $label …"
            mach_tuned, blacklist =
                tune(model, mkspace, X, y; testonly=testonly)

            best_model = fitted_params(mach_tuned).model.best_model
            # We don't log the best fitted params right (i.e. we only log
            # hyperparameters) now because we retrain in `runbest` anyway.
            # best_fitted_params =
            #     fitted_params(mach_tuned).model.best_fitted_params.fitresult

            # Filter out blacklisted fieldnames.
            params_model = filter(
                kv -> kv.first ∉ blacklist,
                Dict(pairs(params(best_model))),
            )

            # Note that for XCSF, we extract and log Julia model params (and not the
            # underlying Python library's params) for now.
            # TODO Log RNG seed
            logartifact(mlf, mlfrun, "best_params.json", json(params_model))

            @info "Finishing run $name_run_final …"
            updaterun(mlf, mlfrun, "FINISHED")
            @info "Finished run $name_run_final."
        end
    end
end

function getoptparams(mlf, label, hash_task)
    mlfexp = getexperiment(mlf, "optparams")
    mlfruns = searchruns(
        mlf,
        mlfexp;
        filter_params=Dict(
            "algorithm.name" => label,
            "task.hash" => hash_task,
        ),
    )

    df = runs_to_df(mlfruns)

    len = nrow(df)
    df = dropmissing(df, "end_time")
    len2 = nrow(df)
    @info "Dropped $(len - len2) unfinished runs."

    df[!, "start_time"] .=
        Dates.unix2datetime.(round.(df.start_time / 1000)) .+
        Millisecond.(df.start_time .% 1000) .+
        # TODO Consider to use TimeZones.jl
        # Add my timezone.
        Hour(2)
    df[!, "end_time"] .=
        passmissing(
            Dates.unix2datetime,
        ).(passmissing(round).(df.end_time / 1000)) .+
        passmissing(Millisecond).(df.end_time .% 1000) .+
        # TODO Consider to use TimeZones.jl
        # Add my timezone.
        Hour(2)

    if length(mlfruns) > 1
        @warn "Ambiguous optparams entries (total of $(length(mlfruns))) in MLflow tracking server, using most recent"
    elseif length(mlfruns) < 1
        error(
            "No optimized hyperparameters for algorithm $label on task $hash_task. Run `optparams` first.",
        )
    end

    row = sort(df, "end_time")[end, :]
    family = getfield(Main, Symbol(row."params.algorithm.family"))

    fname_params = row.artifact_uri * "/best_params.json"
    params = JSON.parsefile(fname_params; dicttype=Dict{Symbol,Any})
    fixparams!(family, params)

    return family, params
end

"""
Run each configured ML algorithms using the best hyperparameters found by
`optparams` on each of given learning tasks. Best hyperparameters are loaded
from mlflow using a combination of ML algorithm label and learning task hash.

# Args

- `fnames`: Name of NPZ files that each containing training data for one
  learning task.

# Options

- `--name-run`: Give the runs performed (one per ML algorithm configured) the
  same specific name.

# Flags

- `--testonly`: Whether to perform very short runs that do not attempt to be
  competitive (useful for testing the pipeline).
"""
@cast function runbest(
    fnames...;
    testonly::Bool=false,
    name_run::Union{Missing,String}=missing,
)
    for fname in fnames
        @info "Starting best-parametrization runs for learning task $fname."
        X, y, hash_task, X_test, y_test = readdata(fname)
        N, DX = nrow(X), ncol(X)

        mlf = getmlf()
        @info "Logging to mlflow tracking URI $(mlf.baseuri)."

        name_exp = "runbest"
        @info "Setting experiment name to $name_exp …"
        mlfexp = getorcreateexperiment(mlf, name_exp)

        for (label, str_family, _, _) in listvariants(N; testonly=testonly)
            @info "Starting run …"
            mlfrun = createrun(mlf, mlfexp; run_name=name_run)
            name_run_final = mlfrun.info.run_name
            @info "Started run $name_run_final with id $(mlfrun.info.run_id)."

            family, params = getoptparams(mlf, label, hash_task)
            model = family(; params...)

            logparam(
                mlf,
                mlfrun,
                Dict(
                    "algorithm.family" => str_family,
                    "algorithm.name" => label,
                    "task.hash" => hash_task,
                    "task.DX" => DX,
                    "task.N" => "N",
                ),
            )

            @info "Transforming training input data …"
            # Note that we cannot use a pipeline here because we need the scaler
            # later.
            scaler = MinMaxScaler()
            mach_scaler = machine(scaler, X)
            MLJ.fit!(mach_scaler)
            X = MLJ.transform(mach_scaler, X)

            @info "Fitting best configuration of $label for task $(string(hash_task; base=16)) …"
            mach = machine(model, X, y)
            MLJ.fit!(mach)

            @info "Computing prediction metrics …"
            y_pred = MLJ.predict_mean(mach, X)
            y_test_pred = MLJ.predict_mean(mach, X_test)

            mae_train = mae(y, y_pred)
            mae_test = mae(y_test, y_test_pred)
            rmse_train = rmse(y, y_pred)
            rmse_test = rmse(y_test, y_test_pred)

            rs = rules(family, fitted_params(mach))
            # We'd have to convert stuff to the MLJ table format if we wanted to use
            # `inverse_transform` directly. I.e. convert `Intervals.Interval`'s
            # bounds etc. to be feature name–based.
            xmin = fitted_params(mach_scaler).xmin
            xmax = fitted_params(mach_scaler).xmax
            intervals =
                inverse_transform_interval.(rs.intervals, Ref(xmin), Ref(xmax))

            mktempdir() do path
                fpath = path * "/rules.jls"
                serialize(fpath, intervals)
                return logartifact(mlf, mlfrun, fpath)
            end

            logmetric.(
                Ref(mlf),
                Ref(mlfrun),
                ["train.mae", "train.rmse", "test.mae", "test.rmse"],
                [mae_train, rmse_train, mae_test, rmse_test],
            )

            rep = report(mach)
            n_iter = rep.n_iter
            map(keys(rep)) do k
                log = getfield(rep, k)
                if log isa AbstractVector && length(log) == n_iter
                    for i in eachindex(log)
                        logmetric(mlf, mlfrun, string(k), log[i]; step=i)
                    end
                elseif log isa AbstractMatrix && size(log, 2) == n_iter
                    for i in eachindex(eachcol(log))
                        logmetric(mlf, mlfrun, string(k), log[:, i]; step=i)
                    end
                end
            end

            @info "Finishing run $name_run_final …"
            updaterun(mlf, mlfrun, "FINISHED")
            @info "Finished run $name_run_final."
        end
    end
end

@main
