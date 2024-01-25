using Infiltrator

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
DT = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0

"""
Fixed parameter values that we do not optimize over.

`n` is the number of training data points available.
"""
function basemodel end

# Include algorithm configurations, parametrizations and hyperparameter ranges.
include(joinpath(@__DIR__, "dt.jl"))
include(joinpath(@__DIR__, "xcsf.jl"))
include(joinpath(@__DIR__, "mga.jl"))

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

function listvariants(N; testonly=false)
    return [
        mkvariant(
            GARegressor,
            :posterior,
            32;
            crossover=false,
            testonly=testonly,
        ),
        mkvariant(DT, N, 1, 70; testonly=testonly),
        mkvariant(XCSFRegressor, N, 200; testonly=testonly),
        mkvariant(XCSFRegressor, N, 400; testonly=testonly),
        mkvariant(XCSFRegressor, N, 800; testonly=testonly),
        mkvariant(XCSFRegressor, N, 1600; testonly=testonly),
        # TODO Deduplicate DT and dt (instead of dt provide `params_fixed` or similar)
        # ("MGA-MAE", "GARegressor", mga_mae, mkspace_mga),
    ]
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

function _optparams(fnames...; testonly::Bool=false, name_run::String="")
    # TODO Random seeding
    # TODO Random seeding for XCSF

    for (i, fname) in enumerate(fnames)
        @info "Starting hyperparameter optimization for learning task $fname."
        @info "This is task $i of $(length(fnames))"
        X, y, hash_task, _, _ = readdata(fname)
        N, DX = nrow(X), ncol(X)

        mlf = getmlf()
        @info "Logging to mlflow tracking URI $(mlf.baseuri)."

        name_exp = "optparams"
        @info "Setting experiment name to $name_exp …"
        mlfexp = getorcreateexperiment(mlf, name_exp)

        for variant in listvariants(N; testonly=testonly)
            @info "Starting run …"
            mlfrun = createrun(
                mlf,
                mlfexp;
                run_name=ifelse(name_run == "", missing, name_run),
            )
            name_run_final = mlfrun.info.run_name
            @info "Started run $name_run_final with id $(mlfrun.info.run_id)."

            logparam(
                mlf,
                mlfrun,
                Dict(
                    "algorithm.family" => string(variant.family),
                    "algorithm.name" => variant.label,
                    "task.hash" => hash_task,
                    "task.DX" => DX,
                    "task.N" => "N",
                ),
            )

            @info "Tuning $(variant.label) …"
            mach_tuned, blacklist =
                tune(variant.model, variant.mkspace, X, y; testonly=testonly)

            history = report(mach_tuned).model.history

            measures_per_fold = getproperty.(history, :per_fold)

            # Since `TunedModel` only supports measures that only depend on `y`
            # and `ypred`, we log GARegressor's fitness using the userextras
            # mechanism.
            userextras_per_fold = getproperty.(history, :userextras)

            for i in eachindex(measures_per_fold)
                if i != nothing
                    logmetric(
                        mlf,
                        mlfrun,
                        "measures",
                        # I don't know why right now but entries in
                        # `measures_per_fold` are wrapped in another array.
                        measures_per_fold[i][1];
                        step=i,
                    )
                end
            end

            for i in eachindex(userextras_per_fold)
                if userextras_per_fold[i] != nothing
                    logmetric(
                        mlf,
                        mlfrun,
                        "userextras",
                        userextras_per_fold[i];
                        step=i,
                    )
                end
            end

            best_model = if variant.model isa GARegressor
                # Determine best_model for `GARegressor` based on `userextras`
                # (where we log the fitness to).
                index_best = argmax(mean.(userextras_per_fold))
                history[index_best].model

            else
                fitted_params(mach_tuned).model.best_model
            end

            # Filter out blacklisted fieldnames.
            params_model = filter(
                kv -> kv.first ∉ blacklist,
                Dict(pairs(params(best_model))),
            )

            # Note that for XCSF, we extract and log Julia model params (and not the
            # underlying Python library's params) for now.
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

function _runbest(
    fnames...;
    seed::Int=0,
    testonly::Bool=false,
    name_run::String="",
)
    for (i, fname) in enumerate(fnames)
        @info "Starting best-parametrization runs for learning task $fname."
        @info "This is task $i of $(length(fnames))"
        X, y, hash_task, X_test, y_test = readdata(fname)
        N, DX = nrow(X), ncol(X)

        mlf = getmlf()
        @info "Logging to mlflow tracking URI $(mlf.baseuri)."

        name_exp = "runbest"
        @info "Setting experiment name to $name_exp …"
        mlfexp = getorcreateexperiment(mlf, name_exp)

        for (label, str_family, _, _) in listvariants(N; testonly=testonly)
            @info "Starting run …"
            mlfrun = createrun(
                mlf,
                mlfexp;
                run_name=ifelse(name_run == "", missing, name_run),
            )
            name_run_final = mlfrun.info.run_name
            @info "Started run $name_run_final with id $(mlfrun.info.run_id)."

            family, params = getoptparams(mlf, label, hash_task)
            model = family(; params...)
            model.rng = seed

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

            logparam(
                mlf,
                mlfrun,
                Dict([
                    ("algorithm.param." * string(k), v) for
                    (k, v) in pairs(params)
                ]),
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
            map(keys(rep)) do k
                log = getfield(rep, k)
                if hasproperty(rep, :n_iter) &&
                   log isa AbstractVector &&
                   length(log) == rep.n_iter
                    for i in eachindex(log)
                        logmetric(mlf, mlfrun, string(k), log[i]; step=i)
                    end
                elseif hasproperty(rep, :n_iter) &&
                       log isa AbstractMatrix &&
                       size(log, 2) == rep.n_iter
                    for i in eachindex(eachcol(log))
                        logmetric(mlf, mlfrun, string(k), log[:, i]; step=i)
                    end
                elseif log isa Real
                    logmetric(mlf, mlfrun, string(k), log)
                end
            end

            @info "Finishing run $name_run_final …"
            updaterun(mlf, mlfrun, "FINISHED")
            @info "Finished run $name_run_final."
        end
    end
end
