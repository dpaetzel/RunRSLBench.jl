using AlgebraOfGraphics
using CSV
using CairoMakie
using ColorSchemes
using DataFrames
using Dates
using KittyTerminalImages
using MLFlowClient
using NPZ
using RSLModels.MLFlowUtils
using Serialization
using StatsBase

AOG = AlgebraOfGraphics

# 2024-01-29 run set (effectless bloat)
# jids = 529675:529772
# 2024-01-30 run set
# jids = 533439:533544 (aborted due to wanting to use more data sets)
# 2024-01-30 22:18 run set (seeds 0-4, aka 1756-.*)
jids = 535054:535501
# 2024-01-31 9:10 run set (seeds 0-4, but only additional high mutation runs,
# aka 1756-highm-.*)
# TODO Merge this with previous
# jids = 537998:538177

const nameformat = r"^([^-]+)-(([^-]*)-)?(\d+)_(\d+)$"

function parse_rname(rname)
    tag, _, subtag, jid, tid = match(nameformat, rname)
    return (; tag=tag, subtag=subtag, jid=jid, tid=tid)
end

function _loadruns(jids)
    url = "http://localhost:5001"
    exp_name = "runbest"
    df_orig = loadruns("runbest"; url=url)
    df1 = deepcopy(df_orig)

    # Only keep runs that match the current format for `run_name`.
    df2 = subset(df1, "run_name" => (n -> occursin.(nameformat, n)))

    df3 = transform(df2, "run_name" => ByRow(parse_rname) => AsTable)

    # Filter for job IDs.
    df4 = subset(df3, "jid" => (jid -> jid .∈ Ref(string.(jids))))

    return df4
end

function duration(df)
    df = dropmissing(df, "duration")

    # Convert an ms duration to human-readable format.
    fromms = passmissing(Dates.canonicalize ∘ Dates.CompoundPeriod)
    # tominutes = passmissing(Dates.value)

    function minutes(ms)
        return Dates.value(ceil(ms, Minute))
    end

    display(
        draw(
            data(df) *
            # mapping("params.algorithm.name", "duration" => minutes) *
            mapping("params.algorithm.family", "duration" => minutes) *
            # visual(Violin),
            visual(BoxPlot),
        ),
    )

    println()

    println("When only considering GARegressor:")
    return display(
        draw(
            data(
                subset(
                    df,
                    "params.algorithm.family" =>
                        (family -> family .== "GARegressor"),
                ),
            ) *
            mapping("params.task.DX", "duration" => minutes) *
            # visual(Violin),
            visual(BoxPlot),
        ),
    )
end

_df = _loadruns(jids)

set_theme!()
update_theme!(theme_latexfonts())
update_theme!(;
    size=(800, 800),
    # palette=(; colors=:seaborn_colorblind),
    palette=(; color=reverse(ColorSchemes.seaborn_colorblind.colors)),
)
set_kitty_config!(:scale, 0.8)

df = deepcopy(_df)

# Unselect columns that we do not require.
df = select(
    df,
    Not([
        "start_time",
        "end_time",
        "lifecycle_stage",
        "experiment_id",
        "run_id",
        "status",
    ]),
)

# mlflow stores these as `BigInt` `String`s.
df[!, "params.task.hash"] .= parse.(BigInt, df[:, "params.task.hash"])

# TODO Adjust n_variants later for 3(?) more variants
n_variants = 7
n_tasks = 60
n_reps = 5
n_runs = n_variants * n_tasks * n_reps
if nrow(df) != n_runs
    @warn "Less runs ($(nrow(df))) than expected ($n_runs)!"
end

# duration(df)

alg_pretty = Dict(
    "DT1-70" => "DT (1–70)",
    "XCSF500" => "XCSF (1–500)",
    "XCSF1000" => "XCSF (1–1000)",
    "MGA32-lnselect-spatialx" => "GA x:spt s:len",
    "MGA32-lnselect-nox" => "GA x:off s:len",
    "MGA32-trnmtselect-spatialx" => "GA x:spt s:trn",
    "MGA32-lnselect-cutsplicex" => "GA x:cut s:len",
    "MGA32-trnmtselect-spatialx-highm" => "GA x:spt s:trn m+",
    "MGA32-lnselect-cutsplicex-highm" => "GA x:cut s:len m+",
    "MGA32-lnselect-spatialx-highm" => "GA x:spt s:len m+",
)

alg_sorter = sorter([
    "DT (1–70)",
    "XCSF (1–500)",
    "XCSF (1–1000)",
    "GA x:spt s:len",
    "GA x:off s:len",
    "GA x:spt s:trn",
    "GA x:cut s:len",
    # "GA x:spt s:trn m+",
    # "GA x:cut s:len m+",
    # "GA x:spt s:len m+",
])

# Rename algorithms to pretty names.
df[:, "params.algorithm.name"] .=
    get.(Ref(alg_pretty), df[:, "params.algorithm.name"], missing)

# Extract task `K` values from `stats` files.
#
# I'm aware that this whole push-to-an-array business is ugly and could be done
# better.
Ks = []
for row in eachrow(df)
    fname = replace(row["params.task.fname"], "data.npz" => "stats.jls")
    hashdf = row["params.task.hash"]

    _stats = deserialize(fname)
    hashfile = BigInt(_stats[:hash])
    # Sanity check. Did the runs in mlflow run on the same data I have locally?
    @assert hashdf == hashfile

    push!(Ks, _stats[:stats][:K])
end
df[!, "params.task.K"] = Ks
@assert all(df[:, "params.task.K"] .> 0) "All should have a " *
                                         "non-zero K entry now"

coloralg =
    mapping(; color="params.algorithm.name" => alg_sorter => "Algorithm")

display(
    draw(
        data(df) *
        mapping(
            "metrics.test.mae";
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
        ) *
        coloralg *
        visual(ECDFPlot);
        # axis=(; xscale=log10),
    ),
)

display(
    draw(
        data(df) *
        mapping(
            "metrics.n_rules";
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
        ) *
        coloralg *
        visual(ECDFPlot);
        # axis=(; xscale=log10),
    ),
)

display(
    draw(
        data(
            subset(
                df,
                "params.algorithm.family" => (f -> f .== "GARegressor"),
            ),
        ) *
        mapping(
            "metrics.test.mae";
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
        ) *
        coloralg *
        visual(ECDFPlot);
        # axis=(; xscale=log10),
    ),
)

display(
    draw(
        data(
            subset(
                df,
                "params.algorithm.family" => (f -> f .== "GARegressor"),
            ),
        ) *
        mapping(
            "metrics.n_rules";
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
        ) *
        coloralg *
        visual(ECDFPlot);
        # axis=(; xscale=log10),
    ),
)

dfga = subset(df, "params.algorithm.family" => (f -> f .== "GARegressor"))

idxs = sample(1:nrow(dfga), 20)
for idx in idxs
    # fitness = Matrix(readcsvartifact.(df.artifact_uri, "log_fitness.csv"))
    _fitness =
        Matrix(readcsvartifact(dfga.artifact_uri[idx], "log_fitness.csv"))

    # Let's sort the fitness row-wise (lowest fitness at the start of each row).
    fitness = deepcopy(_fitness)
    fitness = reduce(vcat, transpose.(sort.(eachrow(fitness))))
    fitness = DataFrame(fitness, :auto)
    fitness[!, "Iteration"] = 1:nrow(fitness)
    fitness =
        stack(fitness; variable_name="Ranked Solution", value_name="Fitness")
    display(
        draw(
            data(fitness) *
            mapping("Iteration", "Fitness"; color="Ranked Solution") *
            visual(Lines),
        ),
    )
    println(
        dfga[
            idx,
            ["params.algorithm.name", "params.task.DX", "params.task.K"],
        ],
    )
    println()
end
