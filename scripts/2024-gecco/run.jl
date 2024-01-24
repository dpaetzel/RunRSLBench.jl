using Comonicon

include(joinpath(@__DIR__, "_run.jl"))

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
    return _optparams(fnames...; testonly=testonly, name_run=name_run)
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
    return _runbest(fnames...; testonly=testonly, name_run=name_run)
end

@main
