using Base64
using TOML
using MLFlowClient
using NPZ
# using TreeParzen
using MLJ
using Tables
using RunRSLBench.Transformers
using RunRSLBench.XCSF
DT = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0


function check_param(dict1, dict2, k; name1 = "dict1", name2 = "dict2")
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
        TOML.parse(read(file, String))
    end

    mlflow_url = "http://localhost:5000"

    username = config["database"]["user"]
    password = config["database"]["password"]
    encoded_credentials = base64encode("$username:$password")
    headers = Dict("Authorization" => "Basic $encoded_credentials")
    return MLFlow(mlflow_url, headers = headers)
end


function mkspace_xcsf(pipe, DX)
    return [
        range(pipe, :(model.e0); lower = 0.01, upper = 0.2, scale = :log),
        # TODO Consider even higher beta values as well?
        range(pipe, :(model.beta); lower = 0.001, upper = 0.3, scale = :log),
        # TODO Consider to use cube size
        range(pipe, :(model.condition_spread_min); lower = 0.01, upper = 0.5),
        range(pipe, :(model.ea_select_size); lower = 0.1, upper = 0.8),
        range(pipe, :(model.ea_p_crossover); lower = 0.0, upper = 1.0),
    ]
end


"""
Given input space dimensionality and a minimum and maximum number of rules,
define the hyperparameter space for DT algorithms.
"""
function mkmkspace_dt(K_min, K_max, N)
    max_depth_min = max(1, ceil(log2(K_min / 2)))
    max_depth_max = max(max_depth_min + 1, ceil(log2(K_max / 2)))

    function mkspace_dt(pipe, DX)
        return [
            range(pipe, :(model.max_depth); lower = max_depth_min, upper = max_depth_max),
            range(
                pipe,
                :(model.min_samples_split);
                lower = ceil(0.001 * N),
                upper = ceil(0.05 * N),
                # scale = :log,
            ),
        ]
    end

    return mkspace_dt
end


function listvariants(N; testonly = false)
    xcsf = XCSFRegressor(;
        # TODO Set higher value
        max_trials = ifelse(testonly, 10, 200000),
        # Note that we do not hyperparameter optimize `pop_size` because that
        # would be unfair. We will instead choose 4 fixed values.
        pop_size = 500,
    )

    dt = DT(;
        min_samples_leaf = 1,
        min_purity_increase = 0,
        # Always consider all features.
        n_subfeatures = 0,
    )

    return [
        # WAIT For Preen fixing the ordering bug (should be #112 in XCSF repo), otherwise cannot set XCSF conditions
        # ("XCSF100", xcsf, mkspace_xcsf),
        # ("XCSF200", xcsf, mkspace_xcsf),
        ("DT2-50", dt, mkmkspace_dt(2, 50, N)),
        ("DT2-100", dt, mkmkspace_dt(2, 100, N)),
    ]
end


fname = "2-2-502-0-0.9-true.data.npz"

# data = npzread(fname)
# X, y = data["X"], data["y"]
# # X_test, y_test = data["X_test"], data["y_test"]


# X = MLJ.table(X)
# X_test = MLJ.table(X_test)


function tune(model, mkspace, X, y)
    # We don't use |> because we have to control the name given to each pipeline
    # stage so we can set hyperparameters properly.
    pipe = Pipeline(scaler = MinMaxScaler(), model = model)

    Xmat = Tables.matrix(X)
    N, DX = size(Xmat)

    pipe_tuned = TunedModel(
        model = pipe,
        resampling = CV(nfolds = 5),
        # TODO Right now TreeParzen does not support nested hyperparameters (see my
        # issue on that).
        # tuning = MLJTreeParzenTuning(),
        tuning = LatinHypercube(; gens = 30),
        # TODO Increase resolution
        # tuning = Grid(; resolution = 2),
        range = mkspace(pipe, DX),
        measure = mae,
        # Number of models to evaluate. Note that without this, LatinHypercube fails
        # with an obscure error message.
        # TODO Increase number of models/make more fair wrt runtime
        n = 100,
    )

    mach_tuned = machine(pipe_tuned, X, y)
    fit!(mach_tuned)

    return mach_tuned
end


function optparams(fname; testonly = false, name_run = missing)
    # TODO Random seeding
    # TODO Random seeding for XCSF

    println("Reading data from $fname …")
    data = npzread(fname)
    X, y = data["X"], data["y"]
    X = MLJ.table(X)
    hash_task = data["hash"]
    println("Read data for task $(string(hash_task; base=16)) from $fname.")

    mlf = getmlf()
    println("Logging to mlflow tracking URI $(mlf.baseuri).")

    name_exp = "optparams"
    println("Setting experiment name to $name_exp …")
    mlfexp = getorcreateexperiment(mlf, name_exp)

    Xmat = Tables.matrix(X)
    N, DX = size(Xmat)

    for (label, model, mkspace) in listvariants(N; testonly = testonly)
        println("Starting run …")
        mlfrun = createrun(mlf, mlfexp; run_name = name_run)
        name_run_final = mlfrun.info.run_name
        println("Started run $name_run_final with id $(mlfrun.info.run_id).")

        logparam(mlf, mlfrun, Dict("algorithm.name" => label, "task.hash" => hash_task))

        println("Tuning $label …")
        mach_tuned = tune(model, mkspace, X, y)

        best_model = fitted_params(mach_tuned).best_model
        best_fitted_params = fitted_params(mach_tuned).best_fitted_params


        # For XCSF, we extract and log Julia model params (and not the
        # underlying Python library's params) for now.
        # params_internal = fitted_params(mach_tuned).best_fitted_params.model.internal_params()
        # params_sklearn = fitted_params(mach_tuned).best_fitted_params.model.get_params()
        # for k in keys(params_internal)
        #     check_param(
        #         params_internal,
        #         params_sklearn,
        #         k;
        #         name1 = "XCSF internal",
        #         name2 = "XCSF sklearn",
        #     )
        # end
        fieldnames_ = fieldnames(typeof(model))
        params_model = Dict(
            Symbol.("algorithm." .* string.(fieldnames_)) .=>
                getproperty.(Ref(best_model.model), fieldnames_),
        )
        # TODO Log RNG seed
        logparam(mlf, mlfrun, params_model)

        println("Finishing run $name_run_final …")
        updaterun(mlf, mlfrun, "FINISHED")
        println("Finished run $name_run_final.")
    end
end

label = "DT2-50"
hash_task = 0xa71de19bb0e26f9a
function getoptparams(label, hash_task)
    mlf = getmlf()
    mlfexp = getexperiment(mlf, "optparams")
    mlfruns = searchruns(
        mlf,
        mlfexp;
        filter_params = Dict("algorithm.name" => label, "task.hash" => hash_task),
    )

    if length(mlfruns) > 1
        error("Ambiguous optparams entries in MLflow tracking server")
    end

    if length(mlfruns) < 1
        error(
            "No optimized hyperparameters for algorithm $label on task $hash_task. Run `optparams` first.",
        )
    end

end


# Get population, NEXT extract rules
