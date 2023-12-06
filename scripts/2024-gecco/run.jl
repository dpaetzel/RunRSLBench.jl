using Base64
using Dates
using DataFrames
using JSON
using TOML
using Missings
using MLFlowClient
using MLJ
using NPZ
using RunRSLBench.XCSF
using RSLModels.Transformers
using RSLModels.MLFlowUtils
using Tables
# using TreeParzen
DT = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0

function nonprimitivetostring(value)
    # todo symbols are dropped this way
    # maybe serialize the julia way after all?
    # but i do not want to “optimize” rng and set it afterwards
    # maybe consider a blacklist of params for each algorithm
    if all(.!isa.(Ref(value), [Real, String, Bool]))
        return missing
    end
    return value
end

function check_param(dict1, dict2, k; name1="dict1", name2="dict2")
    if dict1[k] != dict2[k]
        @error """Mismatch between $name1 and $name2 at param "$k"
        - $name1[$k] = $(dict1[k])
        - $name2[$k] = $(dict2[k])
        """
    end
end

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
            range(pipe, :(model.e0); lower=0.01, upper=0.2, scale=:log),
            # TODO Consider even higher beta values as well?
            range(pipe, :(model.beta); lower=0.001, upper=0.3, scale=:log),
            # TODO Consider to use cube size
            range(pipe, :(model.condition_spread_min); lower=0.01, upper=0.5),
            range(pipe, :(model.ea_select_size); lower=0.1, upper=0.8),
            range(pipe, :(model.ea_p_crossover); lower=0.0, upper=1.0),
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
                    :(model.max_depth);
                    lower=max_depth_min,
                    upper=max_depth_max,
                ),
                range(
                    pipe,
                    :(model.min_samples_split);
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

function fixparams!(::Type{DT}, params)
    return params[:feature_importance] = Symbol(params[:feature_importance])
end

function fixparams!(::Type{XCSFRegressor}, params)
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

    return [
        # WAIT For Preen fixing the ordering bug (should be #112 in XCSF repo), otherwise cannot set XCSF conditions
        ("XCSF100", "XCSFRegressor", xcsf, mkspace_xcsf),
        # ("XCSF200", "XCSF", xcsf, mkspace_xcsf),
        # TODO Deduplicate DT and dt (instead of dt provide `params_fixed` or similar)
        ("DT2-50", "DT", dt, mkmkspace_dt(2, 50, N)),
        ("DT2-100", "DT", dt, mkmkspace_dt(2, 100, N)),
    ]
end

fname = "2-2-502-0-0.9-true.data.npz"

# data = npzread(fname)
# X, y = data["X"], data["y"]
# X = MLJ.table(X)

# X_test, y_test = data["X_test"], data["y_test"]
# X_test = MLJ.table(X_test)

function mkpipe(model)
    return Pipeline(; scaler=MinMaxScaler(), model=model)
end

function tune(model, mkspace, X, y)
    # We don't use |> because we have to control the name given to each pipeline
    # stage so we can set hyperparameters properly.
    pipe = mkpipe(model)

    Xmat = Tables.matrix(X)
    N, DX = size(Xmat)

    sb = mkspace(pipe, DX)
    space = sb.space
    blacklist = sb.blacklist

    pipe_tuned = TunedModel(;
        model=pipe,
        resampling=CV(; nfolds=5),
        # TODO Right now TreeParzen does not support nested hyperparameters (see my
        # issue on that).
        # tuning = MLJTreeParzenTuning(),
        tuning=LatinHypercube(; gens=30),
        # TODO Increase resolution
        # tuning = Grid(; resolution = 2),
        range=space,
        measure=mae,
        # Number of models to evaluate. Note that without this, LatinHypercube fails
        # with an obscure error message.
        # TODO Increase number of models/make more fair wrt runtime
        n=100,
    )

    mach_tuned = machine(pipe_tuned, X, y)
    fit!(mach_tuned)

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

function optparams(fname; testonly=false, name_run=missing)
    # TODO Random seeding
    # TODO Random seeding for XCSF

    X, y, hash_task, _, _ = readdata(fname)

    mlf = getmlf()
    @info "Logging to mlflow tracking URI $(mlf.baseuri)."

    name_exp = "optparams"
    @info "Setting experiment name to $name_exp …"
    mlfexp = getorcreateexperiment(mlf, name_exp)

    Xmat = Tables.matrix(X)
    N, DX = size(Xmat)

    for (label, family, model, mkspace) in listvariants(N; testonly=testonly)
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
        mach_tuned, blacklist = tune(model, mkspace, X, y)

        best_model = fitted_params(mach_tuned).best_model
        best_fitted_params = fitted_params(mach_tuned).best_fitted_params

        # Filter out blacklisted fieldnames.
        params_model = filter(
            kv -> kv.first ∉ blacklist,
            Dict(pairs(params(best_model.model))),
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

label = "DT2-50"
hash_task = 0xa71de19bb0e26f9a
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

function runbest(fname; testonly=false, name_run=missing)
    X, y, hash_task, X_test, y_test = readdata(fname)
    Xmat = Tables.matrix(X)
    N, DX = size(Xmat)

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

        @info "Fitting best configuration of $label for task $(string(hash_task; base=16)) …"
        pipe = mkpipe(model)
        mach = machine(pipe, X, y)
        fit!(mach)

        @info "Computing prediction metrics …"
        y_pred = predict(mach, X)
        y_test_pred = predict(mach, X_test)

        mae_train = mae(y, y_pred)
        mae_test = mae(y_test, y_test_pred)
        rmse_train = rmse(y, y_pred)
        rmse_test = rmse(y_test, y_test_pred)

        logmetric.(
            Ref(mlf),
            Ref(mlfrun),
            ["train.mae", "train.rmse", "test.mae", "test.rmse"],
            [mae_train, rmse_train, mae_test, rmse_test],
        )

        @info "Finishing run $name_run_final …"
        updaterun(mlf, mlfrun, "FINISHED")
        @info "Finished run $name_run_final."
    end
    return nothing
end

# Get population, NEXT extract rules
