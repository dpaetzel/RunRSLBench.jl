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

function graphs(df=nothing)
    df = if df == nothing

        # 2024-01-29 run set (effectless bloat)
        # jids = 529675:529772
        # 2024-01-30 run set
        # jids = 533439:533544 (aborted due to wanting to use more data sets)
        # 2024-01-30 22:18 run set (seeds 0-4, aka 1756-.*, n_iter=1000)
        # _df = _loadruns(535054:535501)
        # 2024-01-31 9:10 run set (seeds 0-4, but only 3 additional high
        # mutation runs, aka 1756-highm-.*, n_iter=1000)
        # Note: One run failed due to Slurm error, the repetition is 538918.
        # Note: _df2 has to be merged with _df to get all 10 algorithms.
        # _df2 = _loadruns(vcat(collect(537998:538177), 538918))
        # 2024-01-31 19:29 run set (seeds 0-4, all MGA only, this time n_iter=2000)
        # Note: Has to be merged with XCSF and DT runs from _df
        # _df3 = _loadruns(539284:539727)

        # 2024-02-01 18:57 run set (seeds 0-4, only randinit MGAs)
        # TODO Check these
        # 541404:541547

        # New approach with longer MGA runs (n_iter=2000).
        #
        # Get MGA runs from:
        #
        # 2024-01-31 19:29 run set (seeds 0-4, all MGA only, this time n_iter=2000)
        _df3 = _loadruns(539284:539727)
        @assert nrow(_df3) == 7 * 60 * 5
        # Get DT and XCSF runs from:
        #
        # 2024-01-30 22:18 run set (seeds 0-4, aka 1756-.*, n_iter=1000)
        _df4 = _loadruns(535054:535501)
        _df5 = subset(
            _df4,
            "params.algorithm.family" =>
                (f -> f .∈ Ref(["DT", "XCSFRegressor"])),
        )
        @assert nrow(_df5) == 3 * 60 * 5

        df = vcat(deepcopy(_df3), deepcopy(_df5))
    else
        df
    end

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

    n_variants = 10
    n_tasks = 60
    n_reps = 5
    n_runs = n_variants * n_tasks * n_reps
    if nrow(df) != n_runs
        @warn "Different number of runs ($(nrow(df))) than expected ($n_runs)!"
    end

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

    # Unselect K=2.
    df = subset(df, "params.task.K" => (k -> k .> 2))

    n_variants = 10
    n_tasks = 60 - 6
    n_reps = 5
    n_runs = n_variants * n_tasks * n_reps
    if nrow(df) != n_runs
        @warn "Different number of runs ($(nrow(df))) than expected ($n_runs)!"
    end

    alg_pretty = Dict(
        "DT1-70" => "DT (1–70)",
        "XCSF500" => "XCSF (1–500)",
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
        "XCSF (1–500)",
        "XCSF (1–1000)",
        "GA x:spt s:len",
        "GA x:off s:len m+",
        "GA x:spt s:trn",
        "GA x:cut s:len",
        "GA x:spt s:trn m+",
        "GA x:cut s:len m+",
        "GA x:spt s:len m+",
    ])

    # Rename algorithms to pretty names.
    df[:, "params.algorithm.name"] .=
        get.(Ref(alg_pretty), df[:, "params.algorithm.name"], missing)

    coloralg =
        mapping(; color="params.algorithm.name" => alg_sorter => "Algorithm")

    plt =
        data(df) *
        mapping(
            "metrics.test.mae";
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
        ) *
        coloralg *
        visual(ECDFPlot)
    fig = draw(plt; facet=(; linkxaxes=:none))
    CairoMakie.save("plots/TestMAEAll.pdf", fig)
    display(fig)

    plt =
        data(df) *
        mapping(
            "metrics.train.mae";
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
        ) *
        coloralg *
        visual(ECDFPlot)
    fig = draw(
        plt;
        # axis=(; xscale=log10),
        facet=(; linkxaxes=:none),
    )
    CairoMakie.save("plots/TrainMAEAll.pdf")
    display(fig)

    plt =
        data(df) *
        mapping(
            "metrics.n_rules";
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
        ) *
        coloralg *
        visual(ECDFPlot)
    fig = draw(
        plt;
        # axis=(; xscale=log10),
        facet=(; linkxaxes=:none),
    )
    display(fig)

    plt =
        data(
            subset(
                df,
                "params.algorithm.family" =>
                    (f -> f .∈ Ref(["GARegressor", "DT"])),
            ),
        ) *
        mapping(
            "metrics.test.mae";
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
        ) *
        coloralg *
        visual(ECDFPlot)
    fig = draw(
        plt;
        # axis=(; xscale=log10),
        facet=(; linkxaxes=:none),
    )
    display(fig)

    plt =
        data(
            subset(
                df,
                "params.algorithm.family" =>
                    (f -> f .∈ Ref(["GARegressor", "DT"])),
            ),
        ) *
        mapping(
            "metrics.n_rules";
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
        ) *
        coloralg *
        visual(ECDFPlot)
    fig = draw(
        plt;
        # axis=(; xscale=log10),
        facet=(; linkxaxes=:none),
    )
    display(fig)

    return df
end

function chkearlystop(df)
    sort(
        combine(
            groupby(
                df,
                ["params.task.DX", "params.task.K", "params.algorithm.name"],
            ),
            "metrics.n_iter" => mean,
        ),
    )

    return nothing
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

function fitnessanalysis()
    # NOTE This should work up to the fact that i need to deserialize (or wait a
    # long time for ssh to generate felitist column) the dfga file (including
    # felitist column).
    dfga = subset(df, "params.algorithm.family" => (f -> f .== "GARegressor"))

    idxs = sample(1:nrow(dfga), 20)
    for idx in idxs
        _fitness = readfitness(dfga.artifact_uri[idx])

        fitness = deepcopy(_fitness)

        display(
            draw(
                data(fitness) *
                mapping("Iteration", "Fitness"; color="Ranked Solution") *
                visual(Lines),
            ),
        )

        display(
            draw(
                data(fitness_elitist) *
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

    function getfitnessofelitists()
        error("Takes long, consider to read jls file instead")
        felitist = readfitnesselitist.(dfga.artifact_uri)
        jidsstr = replace(string(jids), ":" => "-")
        # serialize("$jidsstr-felitist.jls", felitist)
        return felitst
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

    dfga[!, "felitist"] = getproperty.(felitist, :Fitness)
    # Make a backup.
    dfga_orig = deepcopy(dfga)

    dfga = combine(
        groupby(
            dfga,
            ["params.task.DX", "params.task.K", "params.algorithm.name"],
        ),
        "felitist" => fitnessmeanvar => AsTable,
    )
    dfga[!, "fstd_lower"] = dfga.fmean - dfga.fstd
    dfga[!, "fstd_upper"] = dfga.fmean + dfga.fstd

    return display(
        draw(
            data(dfga) *
            coloralg *
            mapping(;
                col="params.task.DX" => nonnumeric,
                row="params.task.K" => nonnumeric,
            ) *
            # (
            #     mapping(
            #         "iteration",
            #         "fmean";
            #     ) * visual(Lines)
            # ) *
            # (mapping("iteration", "fstd_lower", "fstd_upper") * visual(Band)),
            (
                mapping(
                    "iteration",
                    "fmean";
                    lower="fstd_lower",
                    upper="fstd_upper",
                ) * visual(LinesFill)
            );
            facet=(; linkyaxes=:none),
        ),
    )
    # return serialize("$jidsstr-dfga.jls", dfga)
end
