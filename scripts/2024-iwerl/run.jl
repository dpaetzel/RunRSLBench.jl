using Comonicon

include(joinpath(@__DIR__, "_run.jl"))

"""
Optimize hyperparameters of each configured ML algorithms for the given learning
tasks, logging results to mlflow.

# Args

- `label_variant`: TODO Doc
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
    label_variant,
    fnames...;
    testonly::Bool=false,
    name_run::String="",
)
    return _optparams(
        label_variant,
        fnames...;
        testonly=testonly,
        name_run=name_run,
    )
end

"""
Run each configured ML algorithms using the best hyperparameters found by
`optparams` on each of given learning tasks. Best hyperparameters are loaded
from mlflow using a combination of ML algorithm label and learning task hash.

# Args

- `label_variant`: TODO Doc
- `fnames`: Name of NPZ files that each containing training data for one
  learning task.

# Options

- `--seed`: RNG seed to use for initializing the learner.
- `--name-run`: Give the runs performed (one per ML algorithm configured) the
  same specific name.

# Flags

- `--testonly`: Whether to perform very short runs that do not attempt to be
  competitive (useful for testing the pipeline).
"""
@cast function runbest(
    label_variant,
    fnames...;
    seed::Int=0,
    testonly::Bool=false,
    name_run::String="",
)
    return _runbest(
        label_variant,
        fnames...;
        seed=seed,
        testonly=testonly,
        name_run=name_run,
    )
end

@main
