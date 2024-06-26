using Infiltrator

using Base64
using CSV
using Dates
using DataFrames
using JSON
using LibGit2
using Missings
using MLFlowClient
using MLJ
using MLJBase
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
using TOML
DT = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0

# TODO Consolidate logtuningreport and logreport
function logtuningreport(mlf, mlfrun, mach)
    rep = report(mach).model.plotting
    n_iter = length(rep.measurements)

    rep_ = Dict{String,Any}(
        rep.parameter_names .=> eachrow(rep.parameter_values'),
    )
    for (k, v) in rep_
        for i in eachindex(v)
            logmetric(mlf, mlfrun, "tuning.params.$k", v[i]; step=i)
        end
    end

    return nothing
end

"""
Fixed parameter values that we do not optimize over.

`n` is the number of training data points available.
"""
function baseparams end

"""
A configuration to evaluate specified as a label for that configuration and a
set of hyperparameter overrides to apply to another base configuration.
"""
struct Override
    label::String
    paramoverrides::Vector{<:Pair}
end

"""
A configuration (`model`) to optimize hyperparameters for. `mkspace` defines the
ranges for hyperparameter optimization.

`additional` is a list of configuration “deltas” that are to be evaluated as
well (i.e. their hyperparameters are not optimized but the same hyperparameters
as the ones of the “main” configuration are used safe for some being
overwritten, see `Override`).
"""
@kwdef struct Variant
    # Label to use for this variant.
    label::String
    # This variant's algorithm family.
    label_family::String
    # The variant's type. Since this may be less legible, `label_family` is
    # provided for custom shortened string representations.
    type_model::Type
    # Base params to use (i.e. the fixed hyperparameters during hyperparameter
    # optimization).
    params::Dict
    # Ranges for hyperparameter optimization.
    mkspace::Union{Function,Nothing}
    # Given optimized hyperparameters, which other configurations to evaluate
    # using those.
    additional::Vector{Override}
end

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

"""
Log a report (e.g. as obtained by `report(mach)` for `mach isa Machine`) to
mlflow.

Note that this only looks at the highest level of the report and does not
descend into nested `NamedTuple`s or similar.
"""
function logreport(mlf, mlfrun, rep, prefix)
    @info "Logging machine report …"
    map(keys(rep)) do k
        log = getfield(rep, k)
        name_metric = ifelse(prefix == "", string(k), "$prefix.$(string(k))")
        # If a vector with as many entries as there are iterations (assumes
        # `n_iter` is the iteration count), log the vector iteration by
        # iteration.
        if log isa AbstractVector
            # See comment below regarding file-based backend.
            # for i in eachindex(log)
            #     logmetric(mlf, mlfrun, name_metric, log[i]; step=i)
            # end
            mktempdir() do path
                fpath = path * "/$(string(k)).csv"
                CSV.write(fpath, Tables.table(log))
                return logartifact(mlf, mlfrun, fpath)
            end
            # Same as vector but for a matrix (i.e. multiple values per iteration).
        elseif log isa AbstractMatrix
            # While this is nice and all, with the file-based backend (yeah, I
            # know, “don't use that duh”), this is a huge bottleneck. Instead,
            # log it to a file and put that into the artifact store.
            # for i in eachindex(eachcol(log))
            #     logmetric(mlf, mlfrun, name_metric, log[:, i]; step=i)
            # end
            mktempdir() do path
                fpath = path * "/$(string(k)).csv"
                CSV.write(fpath, Tables.table(log'))
                return logartifact(mlf, mlfrun, fpath)
            end
        elseif log isa Real
            logmetric(mlf, mlfrun, name_metric, log)
        else
            @warn "Not logging report field $k, type $(typeof(log)) not " *
                  "supported by mlflow"
        end
    end
end

function listvariants(N; testonly=false)
    return [
        mkvariant(
            GARegressor,
            "lnselect-spatialx",
            32;
            select=:lengthniching,
            crossover=:spatial,
            p_mutate=0.05,
            testonly=testonly,
        ),
        mkvariant(
            GARegressor,
            "lnselect-nox",
            32;
            select=:lengthniching,
            crossover=:off,
            p_mutate=0.4,
            testonly=testonly,
        ),
        mkvariant(
            GARegressor,
            "trnmtselect-spatialx",
            32;
            select=:tournament,
            crossover=:spatial,
            p_mutate=0.05,
            testonly=testonly,
        ),
        mkvariant(
            GARegressor,
            "lnselect-cutsplicex",
            32;
            select=:lengthniching,
            crossover=:cutsplice,
            p_mutate=0.05,
            testonly=testonly,
        ),
        mkvariant(
            GARegressor,
            "lnselect-spatialx-highm",
            32;
            select=:lengthniching,
            crossover=:spatial,
            p_mutate=0.4,
            testonly=testonly,
        ),
        mkvariant(
            GARegressor,
            "lnselect-cutsplicex-highm",
            32;
            select=:lengthniching,
            crossover=:cutsplice,
            p_mutate=0.4,
            testonly=testonly,
        ),
        mkvariant(
            GARegressor,
            "trnmtselect-spatialx-highm",
            32;
            select=:tournament,
            crossover=:spatial,
            p_mutate=0.4,
            testonly=testonly,
        ),
        mkvariant(DT, N, 1, 70; testonly=testonly),
        mkvariant(XCSFRegressor, N, 1000; testonly=testonly),
    ]
end

struct HistoryAdditionsSelection <: MLJTuning.SelectionHeuristic end

function MLJTuning.best(::HistoryAdditionsSelection, history)
    # We score using mean of CV here. I'm not that confident that this is the
    # best choice (e.g. median would be another option).
    scores = mean.(getproperty.(history, :history_additions))
    index_best = argmax(scores)
    return history[index_best]
end

# Let's pirate some types. Julia, please forgive me.
function MLJTuning.supports_heuristic(
    ::LatinHypercube,
    ::HistoryAdditionsSelection,
)
    return true
end

function tune(mlf, mlfrun, type_model, params, mkspace, X, y; testonly=false)
    N, DX = nrow(X), ncol(X)

    model = type_model(; params...)

    space = mkspace(model, DX)

    nfolds = 5
    n_gens_lhc = 30
    n_models = ifelse(testonly, 2, 100)

    selection_heuristic = ifelse(
        model isa GARegressor,
        HistoryAdditionsSelection(),
        NaiveSelection(),
    )

    logparam(
        mlf,
        mlfrun,
        Dict(
            "tuning.nfolds" => nfolds,
            "tuning.n_gens_lhs" => n_gens_lhc,
            "tuning.selection_heuristic" => string(selection_heuristic),
            "tuning.n_models" => n_models,
        ),
    )

    pipe = Pipeline(;
        scaler=MinMaxScaler(),
        model=TunedModel(;
            model=model,
            resampling=CV(; nfolds=nfolds),
            # TODO Right now TreeParzen does not support nested hyperparameters (see my
            # issue on that).
            # tuning = MLJTreeParzenTuning(),
            tuning=LatinHypercube(; gens=n_gens_lhc),
            range=space,
            measure=mae,
            selection_heuristic=selection_heuristic,
            # Number of models to evaluate. Note that without this, LatinHypercube fails
            # with an obscure error message.
            # TODO Increase number of models/make more fair wrt runtime
            n=n_models,
            history_additions=ifelse(
                model isa GARegressor,
                history_additions(model),
                (_, _) -> nothing,
            ),
        ),
    )

    mach_tuned = machine(pipe, X, y)
    MLJ.fit!(mach_tuned; verbosity=1000)

    return mach_tuned
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

function getvariant(label_variant, N; testonly=false)
    variants = listvariants(N; testonly=testonly)
    variant = filter(v -> v.label == label_variant, variants)
    if length(variant) != 1
        error("Unknown or ambiguous variant, check the source code")
    end
    return variant[1]
end

function printvariants(; testonly=false)
    # Since we only want to get the names, we use 0 here.
    # TODO Clean up this variants business, properly parallelize
    variants = getproperty.(listvariants(0; testonly=testonly), :label)
    println("Available variants:")
    for v in variants
        println(v)
    end
end

function _optparams(
    label_variant,
    fnames...;
    testonly::Bool=false,
    name_run::String="",
)
    # TODO Random seeding
    # TODO Random seeding for XCSF

    printvariants(; testonly=testonly)

    for (i, fname) in enumerate(fnames)
        @info "Starting hyperparameter optimization for learning task $fname."
        @info "This is task $i of $(length(fnames))"
        @info "Reading training data …"
        X, y, hash_task, _, _ = readdata(fname)
        N, DX = nrow(X), ncol(X)

        mlf = getmlf()
        @info "Logging to mlflow tracking URI $(mlf.baseuri)."

        name_exp = "optparams"
        @info "Setting experiment name to $name_exp …"
        mlfexp = getorcreateexperiment(mlf, name_exp)

        variant = getvariant(label_variant, N; testonly=testonly)

        if variant.mkspace == nothing
            @info "Tuning for $(variant.label) is disabled. Stopping."
            return
        end

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
                "algorithm.family" => string(variant.label_family),
                "algorithm.name" => variant.label,
                "algorithm.testonly" => testonly,
                "task.hash" => hash_task,
                "task.DX" => DX,
                "task.N" => N,
            ),
        )

        # TODO Use blacklist on variant.params if necessary
        logparam(
            mlf,
            mlfrun,
            Dict([
                ("algorithm.param.fixed." * string(k), v) for
                (k, v) in variant.params
            ]),
        )

        @info "Tuning $(variant.label) …"
        mach_tuned = tune(
            mlf,
            mlfrun,
            variant.type_model,
            variant.params,
            variant.mkspace,
            X,
            y;
            testonly=testonly,
        )

        history = report(mach_tuned).model.history

        measures_per_fold = getproperty.(history, :per_fold)

        # Since `TunedModel` only supports measures that only depend on `y`
        # and `ypred`, we log GARegressor's fitness using the
        # `history_additions` mechanism.
        history_additions_per_fold = getproperty.(history, :history_additions)

        for i in eachindex(measures_per_fold)
            if i != nothing
                logmetric(
                    mlf,
                    mlfrun,
                    "tuning.measures",
                    # I don't know why right now but entries in
                    # `measures_per_fold` are wrapped in another array.
                    measures_per_fold[i][1];
                    step=i,
                )
            end
        end

        for i in eachindex(history_additions_per_fold)
            if history_additions_per_fold[i] != nothing
                logmetric(
                    mlf,
                    mlfrun,
                    "tuning.history_additions",
                    history_additions_per_fold[i];
                    step=i,
                )
            end
        end

        best_model = fitted_params(mach_tuned).model.best_model

        # Cheap sanity check for whether we extracted the correct model
        # using the `history_additions` interface.
        if variant.params isa GARegressor
            # Determine best_model for `GARegressor` based on
            # `history_additions` (where we log the fitness to).
            index_best = argmax(mean.(history_additions_per_fold))
            @assert history[index_best].model == best_model
        end

        rep = report(mach_tuned).model.best_report
        logreport(mlf, mlfrun, rep, "best")

        logtuningreport(mlf, mlfrun, mach_tuned)

        # Filter out blacklisted fieldnames.
        paramsdict = filter(
            kv -> kv.first ∉ blacklist(variant.type_model),
            Dict(pairs(params(best_model))),
        )

        # We log optimized hyperparameters to mlflow directly as well as to a
        # JSON artifact. This has historical reasons, we first logged only to a
        # JSON artifact and added the direct logging later.
        logparam(
            mlf,
            mlfrun,
            Dict([
                ("algorithm.param.best." * string(k), v) for
                (k, v) in paramsdict
            ]),
        )

        # Note that for XCSF, we extract and log Julia model params (and not
        # the underlying Python library's params) for now.
        logartifact(mlf, mlfrun, "best_params.json", json(paramsdict))

        @info "Finishing run $name_run_final …"
        updaterun(mlf, mlfrun, "FINISHED")
        @info "Finished run $name_run_final."
    end
end

function getoptparams(mlf, label, hash_task, testonly)
    mlfexp = getexperiment(mlf, "optparams")
    mlfruns = searchruns(
        mlf,
        mlfexp;
        filter_params=Dict(
            "algorithm.name" => label,
            "task.hash" => hash_task,
            "algorithm.testonly" => testonly,
        ),
    )

    if isempty(mlfruns)
        error(
            "No optimized hyperparameters for algorithm $label on task $hash_task. Run `optparams` first.",
        )
    end

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
            "No optimized hyperparameters for algorithm $label on task $hash_task. Run `optparams` fully first.",
        )
    end

    row = sort(df, "end_time")[end, :]

    fname_params = row.artifact_uri * "/best_params.json"
    paramsdict = JSON.parsefile(fname_params; dicttype=Dict{Symbol,Any})

    return paramsdict
end

function _runbest(
    label_variant,
    fnames...;
    seed::Int=0,
    testonly::Bool=false,
    name_run::String="",
)
    printvariants(; testonly=testonly)
    if testonly
        @warn "Expect weird results, testonly=true"
    end

    git_commit = LibGit2.head(".")
    git_dirty = LibGit2.isdirty(GitRepo("."))

    for (i, fname) in enumerate(fnames)
        @info "Starting best-parametrization runs for learning task $fname."
        @info "This is task $i of $(length(fnames))"
        X, y, hash_task, X_test, y_test = readdata(fname)
        N, DX = nrow(X), ncol(X)

        # Not used right now.
        # @info "Deserializing data generating model …"
        # dgmodel = deserialize(replace(fname, ".data.npz" => ".task.jls"))

        mlf = getmlf()
        @info "Logging to mlflow tracking URI $(mlf.baseuri)."

        name_exp = "runbest"
        @info "Setting experiment name to $name_exp …"
        mlfexp = getorcreateexperiment(mlf, name_exp)

        variant = getvariant(label_variant, N; testonly=testonly)

        paramsdict = if variant.mkspace == nothing
            @info "Hyperparameter tuning is disabled for this algorithm. " *
                  "Not trying to load optimized hyperparameters."
            variant.params
        else
            @info "This algorithm was tuned, trying to load " *
                  "optimized hyperparameters …"
            paramsdict =
                getoptparams(mlf, variant.label, hash_task, testonly)
            fixparams!(variant.type_model, paramsdict)
            paramsdict
        end
        model = variant.type_model(; paramsdict...)
        model.rng = seed

        @info "Algorithm seed is $seed."

        # For some algorithms we only optimize a certain config and then
        # reuse the found parameters for other configs.
        overrides = vcat(
            # Add to the overrides the config that we did hyperparameter
            # optimization for.
            [Override(variant.label, empty([0 => 0]))],
            # The remaining overrides specified.
            variant.additional,
        )

        for override in overrides
            let model = deepcopy(model)
                for (k, v) in override.paramoverrides
                    setproperty!(model, k, v)
                end

                @info "Starting run …"
                mlfrun = createrun(
                    mlf,
                    mlfexp;
                    run_name=ifelse(name_run == "", missing, name_run),
                    # https://github.com/JuliaAI/MLFlowClient.jl/issues/30#issue-1855543109
                    tags=[
                        Dict("key" => "gitcommit", "value" => git_commit),
                        Dict(
                            "key" => "gitdirty",
                            "value" => string(git_dirty),
                        ),
                    ],
                )
                name_run_final = mlfrun.info.run_name
                @info "Started run $name_run_final with id $(mlfrun.info.run_id)."

                @info "Logging hyperparameters to mlflow …"
                logparam(
                    mlf,
                    mlfrun,
                    Dict(
                        "algorithm.family" => variant.label_family,
                        "algorithm.name" => override.label,
                        "algorithm.testonly" => testonly,
                        "algorithm.seed" => seed,
                        "task.fname" => fname,
                        "task.hash" => hash_task,
                        "task.DX" => DX,
                        "task.N" => N,
                    ),
                )

                # TODO Deduplicate with above
                paramsdict = filter(
                    kv -> kv.first ∉ blacklist(variant.type_model),
                    Dict(pairs(params(model))),
                )
                fixparams!(variant.type_model, paramsdict)
                logparam(
                    mlf,
                    mlfrun,
                    Dict([
                        ("algorithm.param." * string(k), v) for
                        (k, v) in paramsdict
                    ]),
                )

                @info "Transforming training input data …"
                # Note that we cannot use a pipeline here because we need the scaler
                # later.
                scaler = MinMaxScaler()
                mach_scaler = machine(scaler, X)
                MLJ.fit!(mach_scaler)
                X = MLJ.transform(mach_scaler, X)

                @info "Fitting best configuration of $(override.label) for " *
                      "task $(string(hash_task; base=16)) …"
                mach = machine(model, X, y)
                MLJ.fit!(mach; verbosity=1000)

                @info "Computing prediction metrics …"
                y_pred = MLJ.predict_mean(mach, X)
                y_test_pred = MLJ.predict_mean(mach, X_test)

                mae_train = mae(y, y_pred)
                mae_test = mae(y_test, y_test_pred)
                rmse_train = rmse(y, y_pred)
                rmse_test = rmse(y_test, y_test_pred)

                rs = rules(variant.type_model, fitted_params(mach))
                # We'd have to convert stuff to the MLJ table format if we wanted to use
                # `inverse_transform` directly. I.e. convert `Intervals.Interval`'s
                # bounds etc. to be feature name–based.
                xmin = fitted_params(mach_scaler).xmin
                xmax = fitted_params(mach_scaler).xmax
                intervals =
                    inverse_transform_interval.(
                        rs.intervals,
                        Ref(xmin),
                        Ref(xmax),
                    )

                mktempdir() do path
                    fpath = path * "/rules.jls"
                    serialize(fpath, intervals)
                    return logartifact(mlf, mlfrun, fpath)
                end

                logmetric.(
                    Ref(mlf),
                    Ref(mlfrun),
                    [
                        "train.mae",
                        "train.rmse",
                        "test.mae",
                        "test.rmse",
                        "n_rules",
                    ],
                    [
                        mae_train,
                        rmse_train,
                        mae_test,
                        rmse_test,
                        length(rs.intervals),
                    ],
                )

                rep = report(mach)
                logreport(mlf, mlfrun, rep, "")

                @info "Finishing run $name_run_final …"
                updaterun(mlf, mlfrun, "FINISHED")
                @info "Finished run $name_run_final."
            end
        end
    end
end
