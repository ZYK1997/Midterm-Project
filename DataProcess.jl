# DataProcess.jl

import JSON

# split(trainData[1])
# filter(s -> length(s) > 0, split(trainData[1], r"\s|\.|,|\(|\)"))

# parseData(trainData[2])

#=
parseData(str) = filter(s -> length(s) > 0,
    split(str, r"\s|\.|,|\?|\(|\)|(<.*>)|\"|;|:|!|\*"))
=#
parseData(str) = filter(s -> length(s) > 0,
    split(str, r"(<.*>)|[^0-9a-zA-Z]"))
# parseData("I'm saying: I love; you!")

function getDict(data)
    dict = Dict()
    tot = 0
    for l in data
        ws = parseData(l)
        for w in ws
            if !haskey(dict, w)
                tot += 1
                dict[w] = tot
            end
        end
    end
    dict
end

# getDict(trainData)

function data2Vec(data, dict)
    map(data) do l
        ws = parseData(l)
        v = Vector{Tuple{Int, Int}}()
        s = Set()
        for w in ws
            if !(w in s)
                push!(s, w)
                push!(v, tuple(dict[w], 1))
            end
        end
        sort!(v)
    end
end
