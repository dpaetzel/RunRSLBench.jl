using Infiltrator

using AlgebraOfGraphics
using CairoMakie
using ColorSchemes
using CSV
using DataFrames
using Dates
using KittyTerminalImages
using MLFlowClient
using NPZ
using RSLModels.MLFlowUtils
using RSLModels.Utils
using ProgressMeter
using Serialization
using StatsBase

const AOG = AlgebraOfGraphics

set_theme!()
update_theme!(theme_latexfonts())
update_theme!(;
    size=(800, 800),
    # palette=(; colors=:seaborn_colorblind),
    palette=(; color=reverse(ColorSchemes.seaborn_colorblind.colors)),
)
set_kitty_config!(:scale, 0.8)

const nameformat = r"^([^-]+)-(([^-]*)-)?(\d+)_(\d+)$"

function parse_rname(rname)
    tag, _, subtag, jid, tid = match(nameformat, rname)
    return (; tag=tag, subtag=subtag, jid=jid, tid=tid)
end

"""
Loads from mlflow tracking server at `url` all available runs whose name
contains (see `nameformat`) one of the given job IDs. Checks whether `n_runs`
runs were loaded.

A Julia serialization–based caching mechanism is in place to account for
possibly slow mlflow response. Files are read from the cache without
verification. If the cache did not contain the required file, a the `DataFrame`
loaded from mlflow is written to the cache (but only if the `n_runs` condition
is fulfilled).
"""
function _loadruns(jids, n_runs; url="http://localhost:5001", max_results=5000)
    dname = "cache"
    h4sh = string(hash(jids); base=16)
    fpath = "cache/$(h4sh)_df5.jls"
    df = if isfile(fpath)
        @info "Found file at $fpath, deserializing …"
        deserialize(fpath)
    else
        @info "Did not find file at $fpath, loading from mlflow …"

        exp_name = "runbest"
        df_orig = loadruns("runbest"; url=url, max_results=max_results)
        df1 = deepcopy(df_orig)

        # Only keep runs that match the current format for `run_name`.
        df2 = subset(df1, "run_name" => (n -> occursin.(nameformat, n)))

        df3 = transform(df2, "run_name" => ByRow(parse_rname) => AsTable)

        # Filter for job IDs.
        df4 = subset(df3, "jid" => (jid -> jid .∈ Ref(string.(jids))))

        # If the expected number of rows was loaded, serialize.
        if nrow(df4) == n_runs
            if !isdir(dname)
                mkdir(dname)
            end
            @info "Serializing to $fpath …"
            serialize(fpath, df4)
        end
        df4
    end

    if nrow(df) != n_runs
        @warn "Found only $(nrow(df)) runs instead of the desired $n_runs."
    end

    return df
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

alg_pretty = Dict(
    "DT1-70" => "DT (1–70)",
    "XCSF1000" => "XCSF (1–1000)",
    "MGA32-lnselect-spatialx" => "GA x:spt s:len",
    "MGA32-lnselect-nox" => "GA x:off s:len m+",
    "MGA32-trnmtselect-spatialx" => "GA x:spt s:trn",
    "MGA32-lnselect-cutsplicex" => "GA x:cut s:len",
    "MGA32-trnmtselect-spatialx-highm" => "GA x:spt s:trn m+",
    "MGA32-lnselect-cutsplicex-highm" => "GA x:cut s:len m+",
    "MGA32-lnselect-spatialx-highm" => "GA x:spt s:len m+",
)

alg_sorter = sorter([
    "DT (1–70)",
    "XCSF (1–1000)",
    "GA x:spt s:len",
    "GA x:spt s:trn",
    "GA x:off s:len m+",
    "GA x:cut s:len",
    "GA x:spt s:trn m+",
    "GA x:cut s:len m+",
    "GA x:spt s:len m+",
])

coloralg =
    mapping(; color="params.algorithm.name" => alg_sorter => "Algorithm")

function _loadruns()
    # Get MGA runs from:
    #
    # 2024-01-31 19:29 run set (seeds 0-4, all MGA only, this time MGA's n_iter=2000)
    _df3 = _loadruns(539284:539727, 7 * 60 * 5)
    # Get DT and XCSF runs from:
    #
    # 2024-01-30 22:18 run set (seeds 0-4, aka 1756-.*, MGA's n_iter=1000)
    _df4 = _loadruns(535054:535501, 7 * 60 * 5)
    _df5 = subset(
        _df4,
        "params.algorithm.family" =>
            (f -> f .∈ Ref(["DT", "XCSFRegressor"])),
    )
    @assert nrow(_df5) == 3 * 60 * 5

    return vcat(deepcopy(_df3), deepcopy(_df5))
end

"""
Reads the data from mlflow, prepares it for plotting and serializes it.
"""
function _prep(df)
    # Only keep algorithm variants that were part of the study.
    df = subset(
        df,
        "params.algorithm.name" => (n -> n .∈ Ref(keys(alg_pretty))),
    )

    @info "Checking and preprocessing data …"

    # Ignore unfinished runs but warn the user.
    df_finished =
        subset(df, "status" => (s -> getproperty.(s, :status) .== "FINISHED"))
    if nrow(df_finished) != nrow(df)
        @warn "Not all runs are finished! Ignoring $(nrow(df) - nrow(df_finished)) unfinished runs."
    end
    df = df_finished

    # mlflow stores these as `BigInt` `String`s.
    df[!, "params.task.hash"] .= parse.(BigInt, df[:, "params.task.hash"])

    @info "Extracting learning task metadata from .stats.jls files …"

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

    # Unselect K=2 (we don't have have data for dimensionalities other than 3 for that).
    df = subset(df, "params.task.K" => (k -> k .> 2))

    @info "Checking number of data points …"

    # 7 MGAs, 1 DT, 1 XCSF.
    n_variants = 7 + 1 + 1
    # 3 dimensionalities, 3 numbers of components, 6 learning tasks each.
    n_tasks = 3 * 3 * 6
    n_reps = 5
    n_runs = n_variants * n_tasks * n_reps
    if nrow(df) != n_runs
        @warn "Different number of runs ($(nrow(df))) than expected ($n_runs)!"
    end

    # Rename algorithms to pretty names.
    df[:, "params.algorithm.name"] .=
        get.(Ref(alg_pretty), df[:, "params.algorithm.name"], missing)

    fname = "2024 GECCO Data"
    @info "Serializing data to $fname{.jls,.csv} …"
    serialize("$fname.jls", df)
    CSV.write(
        "$fname.csv",
        df;
        # We have a few `nothing`s which we transform to `missing`s.
        transform=(col, val) -> something(val, missing),
    )

    return nothing
end

function _graphs(df)
    df = deepcopy(df)

    if !isdir("plots")
        mkdir("plots")
    end

    myecdf =
        mapping(;
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
        ) *
        coloralg *
        visual(ECDFPlot)

    onlygadt = data(
        subset(
            df,
            "params.algorithm.family" =>
                (f -> f .∈ Ref(["GARegressor", "DT"])),
        ),
    )

    onlyga = data(
        subset(df, "params.algorithm.family" => (f -> f .== "GARegressor")),
    )

    gadtxcsf = data(
        subset(
            df,
            "params.algorithm.name" => (
                n ->
                    n .∈ Ref([
                        "DT (1–70)",
                        "GA x:off s:len m+",
                        "XCSF (1–1000)",
                    ])
            ),
        ),
    )

    nrules = mapping("metrics.n_rules" => "Number of Rules")
    testmae = mapping("metrics.test.mae" => "Test MAE")

    plt1 = onlyga * testmae * myecdf
    plt2 = onlyga * nrules * myecdf
    fig = Figure(; size=(1200, 600))
    ag1 = draw!(fig[1, 1:4], plt1; facet=(; linkxaxes=:none))
    legend!(fig[1, 5], ag1)
    ag2 = draw!(fig[1, 6:9], plt2; facet=(; linkxaxes=:none))
    CairoMakie.save("plots/GA.pdf", fig)
    display(fig)

    plt3 = gadtxcsf * testmae * myecdf
    plt4 = gadtxcsf * nrules * myecdf
    fig = Figure(; size=(1200, 600))
    ag3 = draw!(fig[1, 1:4], plt3; facet=(; linkxaxes=:none))
    legend!(fig[1, 5], ag3)
    ag4 = draw!(
        fig[1, 6:9],
        plt4;
        facet=(; linkxaxes=:none),
        axis=(; xscale=log10),
    )
    CairoMakie.save("plots/All.pdf", fig)
    display(fig)

    return df
end

function readfitness(artifact_uri)
    _fitness = Matrix(readcsvartifact(artifact_uri, "log_fitness.csv"))

    # Let's sort the fitness row-wise (lowest fitness at the start of each row)
    # and create a nice `DataFrame`.
    fitness = deepcopy(_fitness)
    fitness = reduce(vcat, transpose.(sort.(eachrow(fitness))))
    fitness = DataFrame(fitness, :auto)
    fitness[!, "Iteration"] = 1:nrow(fitness)
    return fitness =
        stack(fitness; variable_name="Ranked Solution", value_name="Fitness")
end

function readfitnesselitist(artifact_uri)
    fitness = readfitness(artifact_uri)
    return subset(fitness, "Ranked Solution" => (s -> s .== "x32"))
end

function fitnessmeanvar(arrays)
    # Each of the 1000 rows is 30 entries: 6 tasks and 5 reps each.
    uuu = reduce(hcat, arrays)
    m = vec(mean(uuu; dims=2))
    return (;
        iteration=collect(1:length(m)),
        fmean=m,
        fstd=vec(std(uuu; dims=2)),
    )
end

function ratedeltas(array, ds=[1400, 1200, 1000, 800, 600, 400, 200])
    return NamedTuple(
        Symbol.("ratedelta" .* string.(ds)) .=>
            array[2000 .- ds] ./ array[end],
    )
end

function elemwisemean(x...; ds=[1400, 1200, 1000, 800, 600, 400, 200])
    return NamedTuple(
        Symbol.("ratedelta" .* string.(ds)) .=> round.(mean.(x); digits=4),
    )
end

function elemwisestd(x...; ds=[1400, 1200, 1000, 800, 600, 400, 200])
    return NamedTuple(
        Symbol.("ratedelta" .* string.(ds)) .=> round.(std.(x); digits=4),
    )
end

function _prepconv(df)
    df_ga = subset(df, "params.algorithm.family" => (f -> f .== "GARegressor"))
    fitnesses = @showprogress map(readfitnesselitist, df_ga.artifact_uri)
    fname = "2024 GECCO Data Fitnesses.jls"
    @info "Serializing fitnesses to $fname …"
    serialize(fname, fitnesses)
    return nothing
end

function _conv(df, fitnesses)
    df_ga = subset(df, "params.algorithm.family" => (f -> f .== "GARegressor"))
    df_ga[!, "felitist"] = getproperty.(fitnesses, :Fitness)
    _df_ga = deepcopy(df_ga)
    df_ga = deepcopy(_df_ga)
    serialize("2024-02-02-10-22-df_ga.jls", df_ga)

    df_ga = transform(df_ga, "felitist" => ByRow(ratedeltas) => AsTable)

    tablemean = sort(
        combine(
            groupby(
                df_ga,
                ["params.algorithm.name"],
                # ["params.task.DX", "params.task.K", "params.algorithm.name"],
            ),
            r"^ratedelta\d+$" => elemwisemean => AsTable,
        ),
    )

    tablestd = sort(
        combine(
            groupby(
                df_ga,
                ["params.algorithm.name"],
                # ["params.task.DX", "params.task.K", "params.algorithm.name"],
            ),
            r"^ratedelta\d+$" => elemwisestd => AsTable,
        ),
    )

    println("stds:")
    display(tablestd)
    println("means:")
    display(tablemean)
    println(tolatex(tablemean))

    return nothing
end
