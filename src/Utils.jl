function nonprimitivetostring(value)
    # todo symbols are dropped this way
    # maybe serialize the julia way after all?
    # but i do not want to “optimize” rng and set it afterwards
    # maybe consider a blacklist of params for each algorithm
    if all(.!isa.(Ref(value), [Real, String, Bool]))
        return missing
    end
    return value
end

function check_param(dict1, dict2, k; name1="dict1", name2="dict2")
    if dict1[k] != dict2[k]
        @error """Mismatch between $name1 and $name2 at param "$k"
        - $name1[$k] = $(dict1[k])
        - $name2[$k] = $(dict2[k])
        """
    end
end
