using Comonicon

include(joinpath(@__DIR__, "_analyse.jl"))

"""
Read the data for the 2024 GECCO paper from mlflow.

# Args

- `tag`: Suffixed to the output path.
"""
@cast function prep(tag=nothing)
    df = _loadruns()
    _prep(df, tag)
    return 0
end

"""
Plot the graphs used in the 2024 GECCO paper.

# Args

- `fpath`: Path to the `.jls` file (a serialized `DataFrame` containing the
  paper's results; see the `prep` method in this file).
- `tag`: Suffixed to the output path.
"""
@cast function graphs(fpath, tag=nothing)
    df = deserialize(fpath)
    _graphs(df, tag)
    return 0
end

@main
