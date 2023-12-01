using Base64
using TOML
using MLFlowClient
using NPZ
# using TreeParzen
using MLJ
using Tables
using RunRSLBench.Transformers
using RunRSLBench.XCSF


fname = "2-2-502-0-0.9-true.data.npz"

data = npzread(fname)
X, y = data["X"], data["y"]
X_test, y_test = data["X_test"], data["y_test"]


X = MLJ.table(X)
X_test = MLJ.table(X_test)


function testxcsfmodel()
    model = XCSFRegressor(; max_trials = 10)
    fitresult, cache, report = MLJ.fit(model, 0, X, y)
    y_test_pred = MLJ.predict(model, fitresult, X_test)
end


testxcsfmodel()


function getmlf()
    # Create MLFlow instance
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


function testmlflow()

    mlf = getmlf()
    experiment_id = getorcreateexperiment(mlf, "test-julia4")

    exprun = createrun(mlf, experiment_id)

    logparam(mlf, exprun, "letsgo", "woooh")
    logmetric(mlf, exprun, "mae", 0.000)
    logartifact(mlf, exprun, "test.txt", "bli bla blu")


    updaterun(mlf, exprun, "FINISHED")
end


# @pyimport optuna

# function mkobjective(::XCSF, X, y)
#     function objective(trial)
#         max_trials = trial.suggest_float("max_trials", 1, 1000)

#         mach = machine(MinMaxScaler() |> model, X, y)
#         TODO Eval
#         TODO CV
#     end

#     return objective_
# end

test = true

model = XCSFRegressor(;
    # TODO Set higher value
    max_trials = ifelse(test, 10, 200000),
    # Note that we do not hyperparameter optimize `pop_size` because that
    # would be unfair. We will instead choose 4 fixed values.
    pop_size = 500,
)

# We don't use |> because we have to control the name given to each pipeline
# stage so we can set hyperparameters properly.
pipe = Pipeline(scaler = MinMaxScaler(), model = model)

function space_xcsf(DX)
    space = [
        range(pipe, :(model.e0); lower = 0.01, upper = 0.2, scale = :log),
        # TODO Consider even higher beta values as well?
        range(pipe, :(model.beta); lower = 0.001, upper = 0.3, scale = :log),
        # TODO Consider to use cube size
        range(pipe, :(model.condition_spread_min); lower = 0.01, upper = 0.5),
        range(pipe, :(model.ea_select_size); lower = 0.1, upper = 0.8),
        range(pipe, :(model.ea_p_crossover); lower = 0.0, upper = 1.0),
    ]
end

Xmat = Tables.matrix(X)
N, DX = size(Xmat)

space = space_xcsf(DX)

pipe_tuned = TunedModel(
    model = pipe,
    resampling = CV(nfolds = 5),
    # TODO Right now TreeParzen does not support nested hyperparameters (see my
    # issue on that).
    # tuning = MLJTreeParzenTuning(),
    tuning = LatinHypercube(; gens = 30),
    # TODO Increase resolution
    # tuning = Grid(; resolution = 2),
    range = space,
    measure = mae,
    # Number of models to evaluate. Note that without this, LatinHypercube fails
    # with an obscure error message.
    # TODO Increase number of models
    n = 100,
)

mach_tuned = machine(pipe_tuned, X, y)
fit!(mach_tuned)


best_model = fitted_params(mach_tuned).best_model

fieldnames_ = fieldnames(XCSFRegressor)
# TODO Should I extract model params
mparams = Dict(fieldnames_ .=> getproperty.(Ref(best_model.model), fieldnames_))
# or *actual* internal hyperparams (best_model)
params_internal = fitted_params(mach_tuned).best_fitted_params.model.internal_params()
# Or sklearn params
params_sklearn = fitted_params(mach_tuned).best_fitted_params.model.get_params()

function check_param(dict1, dict2, k; name1 = "dict1", name2 = "dict2")
    if dict1[k] != dict2[k]
        @error """Mismatch between $name1 and $name2 at param "$k"
        - $name1[$k] = $(dict1[k])
        - $name2[$k] = $(dict2[k])
        """
    end
end

for k in keys(params_internal)
    check_param(
        params_internal,
        params_sklearn,
        k;
        name1 = "XCSF internal",
        name2 = "XCSF sklearn",
    )
end


# Get population, NEXT extract rules
fitted_params(mach_tuned).best_fitted_params.model.json()

mlf = getmlf()
experiment_id = getorcreateexperiment(mlf, "test-julia4")
exprun = createrun(mlf, experiment_id)
logparam(mlf, exprun, Dict("algorithm" => "XCSF"))
mparams_ = Dict(["model." * string(k) => v for (k, v) in mparams])
logparam(mlf, exprun, mparams_)
