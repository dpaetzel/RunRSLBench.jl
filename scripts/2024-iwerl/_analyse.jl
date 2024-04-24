using Infiltrator

using AlgebraOfGraphics
using CairoMakie
using ColorSchemes
using CSV
using DataFrames
using Dates
using KittyTerminalImages
using Makie
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

# function Makie.inverse_transform(::typeof(exp))
#     return log
# end

# function Makie.defaultlimits(::typeof(exp))
#     return (0.0, 1.0)
# end

# function Makie.defined_interval(::typeof(exp))
#     return -Inf .. Inf
# end

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
function _prep(df, tag=nothing)
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

    # Unselect K=2 (we don't have data for dimensionalities other than 3 for that).
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

    fname = isnothing(tag) ? "2024 IWERL Data" : "2024 IWERL Data $tag"
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

function _mse(df, tag=nothing)
    df_ga = subset(df, "params.algorithm.family" => (f -> f .== "GARegressor"))
    uiae = combine(
        groupby(
            df_ga,
            [
                "params.task.DX",
                "params.task.K",
                "params.task.hash",
                "params.algorithm.name",
            ],
        ),
        "metrics.train.rmse" => mean ∘ (x -> x .^ 2) => "mse_mean",
        "metrics.train.rmse" => median ∘ (x -> x .^ 2) => "mse_median",
        "metrics.train.rmse" => minimum ∘ (x -> x .^ 2) => "mse_min",
        "metrics.train.rmse" => maximum ∘ (x -> x .^ 2) => "mse_max",
    )

    uiae_median = combine(
        groupby(uiae, ["params.task.DX", "params.task.K", "params.task.hash"]),
    ) do sdf
        return sdf[argmax(sdf[:, "mse_median"]), :]
    end

    uiae_mean = combine(
        groupby(uiae, ["params.task.DX", "params.task.K", "params.task.hash"]),
    ) do sdf
        return sdf[argmax(sdf[:, "mse_mean"]), :]
    end

    count_highest_mean =
        DataFrame(countmap(uiae_mean[:, "params.algorithm.name"]))
    count_highest_median =
        DataFrame(countmap(uiae_median[:, "params.algorithm.name"]))

    count_highest_mean = stack(count_highest_mean, All())
    count_highest_median = stack(count_highest_median, All();)
    sort!(count_highest_mean, :value; rev=true)
    sort!(count_highest_median, :value; rev=true)
    rename!(
        count_highest_mean,
        :variable => "Variant",
        :value => "Highest Mean MSE Count",
    )
    rename!(
        count_highest_median,
        :variable => "Variant",
        :value => "Highest Median MSE Count",
    )

    println(
        "On how many tasks had each variant the highest mean train " *
        "MSE (pooling over reps)?",
    )
    display(count_highest_mean)
    println(
        "On how many tasks had each variant the highest median train " *
        "MSE (pooling over reps)?",
    )
    display(count_highest_median)

    df_ = DataFrame()
    df_[!, :Variant] = unique(df_ga[:, "params.algorithm.name"])
    df_ = outerjoin(df_, count_highest_mean; on=:Variant)
    df_ = outerjoin(df_, count_highest_median; on=:Variant)

    # Replace missing values with 0.
    df_ = coalesce.(df_, 0)

    sort!(df_, "Highest Mean MSE Count"; rev=true)

    print(tolatex(df_))

    return nothing
end

function tolatex(df)
    return join(names(df), " & ") *
           "\\\\\n" *
           join(join.(Vector.(eachrow(df)), "  &  "), "\\\\\n") *
           "\\\\"
end

function _graphs(df, tag=nothing)
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
    testmse = mapping("metrics.test.rmse" => "Test RMSE")
    trainmae = mapping("metrics.train.mae" => "Train MAE")
    trainmse = mapping("metrics.train.rmse" => "Train RMSE")

    plt1 = onlyga * testmae * myecdf
    plt2 = onlyga * nrules * myecdf
    fig = Figure(; size=(1200, 600))
    ag1 = draw!(
        fig[1, 1:4],
        plt1;
        facet=(; linkxaxes=:none),
        axis=(; ylabel="Density", xticks=LinearTicks(3)),
    )
    legend!(fig[1, 5], ag1)
    ag2 = draw!(
        fig[1, 6:9],
        plt2;
        facet=(; linkxaxes=:none),
        axis=(; ylabel="Density", xticks=LinearTicks(3)),
    )
    fname =
        isnothing(tag) ? "IWERL-GA Test MAE.pdf" : "IWERL-GA Test MAE $tag.pdf"
    CairoMakie.save("plots/$fname", fig)
    println()
    display(fig)

    plt1 = onlyga * trainmae * myecdf
    plt2 = onlyga * nrules * myecdf
    fig = Figure(; size=(1200, 600))
    ag1 = draw!(
        fig[1, 1:4],
        plt1;
        facet=(; linkxaxes=:none),
        axis=(; ylabel="Density", xticks=LinearTicks(3)),
    )
    legend!(fig[1, 5], ag1)
    ag2 = draw!(
        fig[1, 6:9],
        plt2;
        facet=(; linkxaxes=:none),
        axis=(; ylabel="Density", xticks=LinearTicks(3)),
    )
    fname =
        isnothing(tag) ? "IWERL-GA Train MAE.pdf" :
        "IWERL-GA Train MAE $tag.pdf"
    CairoMakie.save("plots/$fname", fig)
    println()
    display(fig)

    plt1 = onlyga * trainmse * myecdf
    plt2 = onlyga * testmse * myecdf
    fig = Figure(; size=(1200, 600))
    ag1 = draw!(
        fig[1, 1:4],
        plt1;
        facet=(; linkxaxes=:none),
        axis=(; ylabel="Density", xticks=LinearTicks(3)),
    )
    legend!(fig[1, 5], ag1)
    ag2 = draw!(
        fig[1, 6:9],
        plt2;
        facet=(; linkxaxes=:none),
        axis=(; ylabel="Density", xticks=LinearTicks(3)),
    )
    fname =
        isnothing(tag) ? "IWERL-GA Train Test MSE.pdf" :
        "IWERL-GA Train Test MSE $tag.pdf"
    CairoMakie.save("plots/$fname", fig)
    println()
    display(fig)

    plt3 = gadtxcsf * testmae * myecdf
    plt4 = gadtxcsf * nrules * myecdf
    fig = Figure(; size=(1200, 600))
    ag3 = draw!(
        fig[1, 1:4],
        plt3;
        facet=(; linkxaxes=:none),
        axis=(; ylabel="Density"),
    )
    legend!(fig[1, 5], ag3)
    ag4 = draw!(
        fig[1, 6:9],
        plt4;
        facet=(; linkxaxes=:none),
        axis=(; xscale=log10, ylabel="Density"),
    )
    fname = isnothing(tag) ? "IWERL-All.pdf" : "IWERL-All $tag.pdf"
    CairoMakie.save("plots/$fname", fig)
    println()
    display(fig)

    # plt2 = onlyga * testmse * myecdf
    plt1 = onlyga * trainmse * myecdf
    fig = Figure(; size=(800, 600))
    ag1 = draw!(
        fig[1, 1:4],
        plt1;
        facet=(; linkxaxes=:none),
        axis=(; ylabel="Density", xticks=LinearTicks(3)),
    )
    legend!(fig[1, 5], ag1)
    fname =
        isnothing(tag) ? "IWERL-GA Train MSE.pdf" :
        "IWERL-GA Train MSE $tag.pdf"
    CairoMakie.save("plots/$fname", fig)
    println()
    display(fig)

    return df
end

"""
Read fitness data for a single run from the given artifact URI.
"""
function readlength(artifact_uri; usecache=true)
    dname = "cache_readlength"
    if !isdir(dname)
        mkdir(dname)
    end
    h4sh = string(hash(artifact_uri); base=16)
    fpath = "$dname/$(h4sh).jls"
    if isfile(fpath) && usecache
        @info "Found file at $fpath, deserializing …"
        return deserialize(fpath)
    else
        _len = Matrix(readcsvartifact(artifact_uri, "log_length.csv"))

        len = deepcopy(_len)
        len = DataFrame(len, :auto)

        len[!, "Iteration"] = 1:nrow(len)
        len = stack(len; variable_name="Solution", value_name="Length")

        if usecache
            serialize(fpath, len)
        end

        return len
    end
end

"""
Read fitness data for a single run from the given artifact URI.
"""
function readfitness(artifact_uri; usecache=true)
    dname = "cache_readfitness"
    if !isdir(dname)
        mkdir(dname)
    end
    # TODO Use experiment/run id directly here
    h4sh = string(hash(artifact_uri); base=16)
    fpath = "$dname/$(h4sh).jls"
    if isfile(fpath) && usecache
        @info "Found file at $fpath, deserializing …"
        return deserialize(fpath)
    else
        _fitness = Matrix(readcsvartifact(artifact_uri, "log_fitness.csv"))

        fitness = deepcopy(_fitness)
        fitness = DataFrame(fitness, :auto)
        fitness[!, "Iteration"] = 1:nrow(fitness)
        fitness =
            stack(fitness; variable_name="Solution", value_name="Fitness")

        if usecache
            serialize(fpath, fitness)
        end

        return fitness
    end
end

# """
# Read fitness data for a single run from the given artifact URI.
# """
# function readfitness_sorted(artifact_uri; usecache=true)
#     dname = "cache_readfitness_sorted"
#     if !isdir(dname)
#         mkdir(dname)
#     end
#     # TODO Use experiment/run id directly here
#     h4sh = string(hash(artifact_uri); base=16)
#     fpath = "$dname/$(h4sh).jls"
#     if isfile(fpath) && usecache
#         @info "Found file at $fpath, deserializing …"
#         return deserialize(fpath)
#     else
#         _fitness = Matrix(readcsvartifact(artifact_uri, "log_fitness.csv"))

#         # Let's sort the fitness row-wise (lowest fitness at the start of each row)
#         # and create a nice `DataFrame`.
#         fitness = deepcopy(_fitness)
#         fitness = reduce(vcat, transpose.(sort.(eachrow(fitness))))
#         fitness = DataFrame(fitness, :auto)
#         fitness[!, "Iteration"] = 1:nrow(fitness)
#         fitness = stack(
#             fitness;
#             variable_name="Ranked Solution",
#             value_name="Fitness",
#         )

#         if usecache
#             serialize(fpath, fitness)
#         end

#         return fitness
#     end
# end

# function readfitnesselitist(artifact_uri)
#     fitness = readfitness_sorted(artifact_uri)
#     # Since we sorted the fitness in `readfitness`, we can extract elitist
#     # fitness by looking at the last fitness. Since we have a population size of
#     # 32 that fitness is at position 32.
#     return subset(fitness, "Ranked Solution" => (s -> s .== "x32"))
# end

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

function _prepconv(df, tag=nothing)
    df_ga = subset(df, "params.algorithm.family" => (f -> f .== "GARegressor"))
    fitnesses = @showprogress map(readfitness, df_ga.artifact_uri)
    fname =
        isnothing(tag) ? "2024 IWERL Data Fitness.jls" :
        "2024 IWERL Data Fitness $tag.jls"
    @info "Serializing fitnesses to $fname …"
    serialize(fname, fitnesses)
    return nothing
end

"""
df : Main DataFrame with all run data.
fitnesses : Elitist fitness histories.
"""
function _conv(df, fitnesses, tag)
    # Restrict to GARegressor.
    df_ga = subset(df, "params.algorithm.family" => (f -> f .== "GARegressor"))

    # Put elitist fitness history from fitnesses into main DataFrame.
    #
    # We can get the entire row (required so we can later use the "Solution" key
    # to extract the lenght) of the maximum fitness entry for each iteration
    # like so:
    df_ga[!, "felitist"] = [
        combine(groupby(fitness, "Iteration")) do sdf
            return sdf[argmax(sdf.Fitness), :]
        end for fitness in fitnesses
    ]

    # Now we can extract the fitness arrays from the fitness DataFrames
    # (DataFrames were required for the innerjoin).
    df_ga[!, "felitist"] = getproperty.(df_ga.felitist, :Fitness)

    serialize(".df_ga1.jls", df_ga)
    # _df_ga = deepcopy(df_ga)

    # Fill shorter runs by repeating their last value.
    len_max = maximum(length.(df_ga.felitist))
    for v in df_ga.felitist
        if length(v) .< len_max
            val_last = v[end]
            len = length(v)
            @info "Filling converged run of length $len …"
            resize!(v, len_max)
            v[(len + 1):end] .= val_last
        end
    end

    # Let's save this for now so we don't have to redo the steps.
    serialize(".df_ga2.jls", df_ga)
    # _df_ga = deepcopy(df_ga)

    # array_fitness = reduce(vcat, df_ga.felitist)

    # Number of runs is used to check whether flattening with respect to fitness
    # values worked as expected.
    nruns = nrow(df_ga)
    len_max = maximum(length.(df_ga.felitist))

    # Add generation indices so that we can remember which fitness value
    # occurred in which generation.
    df_ga[!, "generation"] .= Ref(1:len_max)
    # Now flatten with respect to generation and fitness elitist value
    # iterables.
    df_ga = flatten(df_ga, ["generation", "felitist"])
    # Check whether this did what we expected in terms of the number of rows.
    @assert nrow(df_ga) == nruns * len_max

    # Last done at 18:04 2024-04-12.
    serialize(".df_ga4.jls", df_ga)
    # _df_ga = deepcopy(df_ga)

    # Normalize fitness values with min and max values observed per task.
    df_ga = transform!(
        groupby(df_ga, ["params.task.hash"]),
        "felitist" => minimum => "fmin",
        "felitist" => maximum => "fmax",
    )

    df_ga = transform!(
        groupby(df_ga, ["params.task.hash"]),
        ["felitist", "fmin", "fmax"] =>
            (
                (f, fmin, fmax) -> @. (f - fmin) / (fmax - fmin)
            ) => "felitist_norm",
    )

    rename!(df_ga, "generation" => "Generation")

    df_ga_small = subset(df_ga, "Generation" => (g -> @. g % 25 == 0))

    # TODO Consider to look at population fitness diversity
    # TODO Consider to look at population length diversity

    quantlower = 0.1
    quantupper = 0.9
    uiae = combine(
        groupby(
            df_ga_small,
            [
                "params.task.DX",
                "params.task.K",
                # "params.task.hash",
                "params.algorithm.name",
                "Generation",
            ],
        ),
        "felitist_norm" => mean,
        "felitist_norm" => std,
        "felitist_norm" => median,
        "felitist_norm" =>
            (
                f -> quantile(f, quantlower)
            ) => "felitist_norm_quantile$quantlower",
        "felitist_norm" =>
            (
                f -> quantile(f, quantupper)
            ) => "felitist_norm_quantile$quantupper",
        "felitist_norm" => minimum,
        "felitist_norm" => maximum,
    )

    uiae = transform(
        uiae,
        ["felitist_norm_mean", "felitist_norm_std"] =>
            ((m, s) -> m - s) => "felitist_norm_std_lower",
        ["felitist_norm_mean", "felitist_norm_std"] =>
            ((m, s) -> m + s) => "felitist_norm_std_upper",
    )

    plt =
        data(uiae) *
        # data(uiae_) *
        (
            mapping(
                "Generation",
                "felitist_norm_quantile$quantlower",
                "felitist_norm_quantile$quantupper",
            ) * visual(Band; alpha=0.2) +
            # mapping("Generation", "felitist_norm_mean") *
            mapping("Generation", "felitist_norm_median") *
            visual(Lines; linewidth=1.7)
            # mapping("Generation", "felitist_norm_mean", "felitist_norm_std") *
            # visual(Errorbars)
        ) *
        mapping(;
            # layout="params.task.hash" => nonnumeric,
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
            # marker="params.algorithm.name" => nonnumeric,
            linestyle="params.algorithm.name" => alg_sorter,
        ) *
        coloralg

    fig = Figure(; size=(1200, 600))
    ag = draw!(
        fig[1, 1:4],
        plt;
        facet=(; linkyaxes=:none),
        axis=(; ylabel="Normalized elitist fitness"),
    )
    # legend!(fig[1, 5], ag; merge=true)
    # legend!(fig[1, 5], ag; unique=true)
    # Inspired by
    # https://github.com/MakieOrg/AlgebraOfGraphics.jl/issues/434#issue-1434646626
    legend = AOG.compute_legend(ag)
    # legend[1] are the legend indicators.
    # legend[2] are the legend labels.
    # first have to sort

    # Problem: legend[1][1] is only the linestyles while legend[1][2] is only
    # the colors.
    #
    # Idea: Set legend[1][2]'s linestyles to the one of legend[1][1].
    for (e1, e2) in zip(legend[1][1], legend[1][2])
        deleteat!(e2, 1)
        # e2[2].attributes[:linestyle] = e1[2].attributes[:linestyle]
        e2[1].attributes[:linestyle] = e1[1].attributes[:linestyle]
    end
    # Remove the now unneeded indicators legend[1][1] as well as the
    # corresponding labels legend[2][1] as well as the legend heading
    # legend[3][1].
    deleteat!(legend[1], 1)
    deleteat!(legend[2], 1)
    deleteat!(legend[3], 1)

    Legend(fig[1, 5], legend...)

    fname = isnothing(tag) ? "IWERL-Fitness.pdf" : "IWERL-Fitness $tag.pdf"
    CairoMakie.save("plots/$fname", fig)
    display(fig)

    return nothing
end

function todo()
    prob = subset(
        df_ga,
        "params.task.DX" => (x -> x .== "3"),
        "params.task.K" => (x -> x .== 4),
    )

    quantlower = 0.1
    quantupper = 0.9
    uiae_prob = combine(
        groupby(
            prob,
            [
                "params.task.DX",
                "params.task.K",
                "params.task.hash",
                "params.algorithm.name",
                "Generation",
            ],
        ),
        "felitist_norm" => mean,
        "felitist_norm" => std,
        "felitist_norm" => median,
        "felitist_norm" =>
            (
                f -> quantile(f, quantlower)
            ) => "felitist_norm_quantile$quantlower",
        "felitist_norm" =>
            (
                f -> quantile(f, quantupper)
            ) => "felitist_norm_quantile$quantupper",
        "felitist_norm" => minimum,
        "felitist_norm" => maximum,
    )

    plt =
        data(uiae_prob) *
        # data(uiae_) *
        (
            mapping(
                "Generation",
                "felitist_norm_quantile$quantlower",
                "felitist_norm_quantile$quantupper",
            ) * visual(Band; alpha=0.2) +
            # mapping("Generation", "felitist_norm_mean") *
            mapping("Generation", "felitist_norm_median") *
            visual(Lines; linewidth=1.7)
            # mapping("Generation", "felitist_norm_mean", "felitist_norm_std") *
            # visual(Errorbars)
        ) *
        mapping(;
            # layout="params.task.hash" => nonnumeric,
            col="params.task.DX" => nonnumeric,
            row="params.task.K" => nonnumeric,
            # marker="params.task.hash" => nonnumeric,
            linestyle="params.task.hash" => nonnumeric,
        ) *
        coloralg

    fig = Figure(; size=(1200, 600))
    ag = draw!(
        fig[1, 1:4],
        plt;
        facet=(; linkyaxes=:none),
        axis=(; ylabel="Normalized elitist fitness"),
    )
    # legend!(fig[1, 5], ag; merge=true)
    # legend!(fig[1, 5], ag; unique=true)
    # Inspired by
    # https://github.com/MakieOrg/AlgebraOfGraphics.jl/issues/434#issue-1434646626
    legend = AOG.compute_legend(ag)
    # legend[1] are the legend indicators.
    # legend[2] are the legend labels.
    # first have to sort

    # Problem: legend[1][1] is only the linestyles while legend[1][2] is only
    # the colors.
    #
    # Idea: Set legend[1][2]'s linestyles to the one of legend[1][1].
    for (e1, e2) in zip(legend[1][1], legend[1][2])
        deleteat!(e2, 1)
        # e2[2].attributes[:linestyle] = e1[2].attributes[:linestyle]
        e2[1].attributes[:linestyle] = e1[1].attributes[:linestyle]
    end
    # Remove the now unneeded indicators legend[1][1] as well as the
    # corresponding labels legend[2][1] as well as the legend heading
    # legend[3][1].
    deleteat!(legend[1], 1)
    deleteat!(legend[2], 1)
    deleteat!(legend[3], 1)

    Legend(fig[1, 5], legend...)

    fname = isnothing(tag) ? "IWERL-Fitness.pdf" : "IWERL-Fitness $tag.pdf"
    CairoMakie.save("plots/$fname", fig)
    display(fig)

    return nothing
end
