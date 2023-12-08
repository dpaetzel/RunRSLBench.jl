module RuleSupport

using JSON
using RunRSLBench.XCSF
using RSLModels.Intervals
using DecisionTree
using MLJ

DT = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0

export rules, inverse_transform_interval

function inverse_transform_interval(interval::Intervals.Interval, xmin, xmax)
    # We'd have to convert stuff to the MLJ table format if we wanted to use
    # `inverse_transform` directly. I.e. convert `Intervals.Interval`'s bounds
    # etc. to be feature name–based.
    lbound = interval.lbound .* (xmax .- xmin) .+ xmin
    ubound = interval.ubound .* (xmax .- xmin) .+ xmin
    return Intervals.Interval(
        lbound,
        ubound;
        lopen=interval.lopen,
        uopen=interval.uopen,
    )
end

"""
    rules(type, fitresult)

Extract interval-based rules from the `fitresult` assuming that it was generated
by a model of type `type`.

The rules are returned in a named tuple containing at least the name `intervals`
for the rules' conditions.
"""
function rules end

function rules(::Type{XCSFRegressor}, fitresult)
    pyxcsf = fitresult
    return_condition = true
    return_action = true
    return_prediction = true
    json_pop = pyxcsf.json(return_condition, return_action, return_prediction)
    rules_ = JSON.parse(json_pop)["classifiers"]

    lowers = []
    uppers = []
    preds = []

    for rule in rules_
        cond = rule["condition"]

        if cond["type"] == "hyperrectangle_csr"
            center = convert(Array, rule["condition"]["center"])
            spread = convert(Array, rule["condition"]["spread"])
            lower = center - spread
            upper = center + spread
        else
            error("bounds_ only exists for hyperrectangular CSR conditions")
        end

        pred = rule["prediction"]["prediction"][1]

        push!(lowers, lower)
        push!(uppers, upper)
        push!(preds, pred)
    end

    DX = length(lowers[1])
    intervals = collect(
        map(
            (l, u) -> Intervals.Interval(
                l,
                u;
                # Preen's implementation's center spread intervals are fully
                # open.
                lopen=fill(true, DX),
                uopen=fill(true, DX),
            ),
            lowers,
            uppers,
        ),
    )
    return (intervals=intervals, predictions=preds)
end

_leq_threshold = "<="
_gt_threshold = ">"

function addrules!(out::Vector, node, featids, rels, thresholds, DX::Int)
    if node isa DecisionTree.Node
        featid = node.featid
        featval = node.featval

        featids_ = vcat(featids, featid)
        thresholds_ = vcat(thresholds, featval)

        addrules!(
            out,
            node.left,
            featids_,
            vcat(rels, _leq_threshold),
            thresholds_,
            DX,
        )
        addrules!(
            out,
            node.right,
            featids_,
            vcat(rels, _gt_threshold),
            thresholds_,
            DX,
        )
    else
        lowers, uppers = fill(-Inf, DX), fill(Inf, DX)

        for d in 1:DX
            idx = findall(d .== featids)
            rels_ = rels[idx]
            thresholds_ = thresholds[idx]

            if !any(rels_ .== _gt_threshold)
                lowers[d] = -Inf
            else
                lowers[d] = max(thresholds_[rels_ .== _gt_threshold]...)
            end

            if !any(rels_ .== _leq_threshold)
                uppers[d] = Inf
            else
                uppers[d] = min(thresholds_[rels_ .== _leq_threshold]...)
            end
        end

        push!(
            out,
            Intervals.Interval(
                lowers,
                uppers;
                # Note that this actually creates intervals that include `Inf`
                # which is non-standard in maths but I guess it does not matter
                # that much.
                lopen=fill(true, DX),
                uopen=fill(false, DX),
            ),
        )
    end
end

function rules(::Type{DT}, fitresult)
    tree = fitresult.tree
    feature_names = tree.info.featurenames
    out = Intervals.Interval[]
    addrules!(out, tree.node, [], [], [], length(feature_names))
    # TODO Consider to also export predictions
    return (intervals=out,)
end

end
