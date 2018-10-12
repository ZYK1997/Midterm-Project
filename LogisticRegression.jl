"""
Logistic Regression
"""

#=
"""
    logisticRegression(examples, η; steps)

2 labels, y can only be 0 or 1.
`η` is the step length.
`steps` is the number of steps.
"""
function logisticRegression(examples, η::Float64; steps = typemax(Int))
    for (x, y) in examples
        push!(x, 1)
    end
    N = length(examples)
    M = length(examples[1][1])
    # w = rand(M)
    w = rand(M)

    h(x) = 1 / (1 + exp(-w' * x))
    loss(examples) = sum(examples) do (x, y)
        tmp = h(x)
         - y * log2(tmp) - (1 - y) * log2(1 - tmp)
    end

    cnt = 0
    loss0 = loss(examples)
    while true
        a = map(examples) do (x, y)
            tmp = w' * x
            1 / (1 + exp(-tmp)) - y
        end

        for j = 1:M
            s = 0
            for i = 1:N
                s += a[i] * examples[i][1][j]
            end
            w[j] -= η * s
        end
        cnt += 1
        loss1 = loss(examples)
        @show (loss0, loss1)
        if cnt == step || (loss0 - loss1) / loss0 < 0.0001
            break
        end
        loss0 = loss1
    end
    w
end
=#

logisticRegression([([0, 1], 0), ([1, 1], 0), ([3, 3], 1), ([4, 3], 1)], 0.1, steps = 1000)

function logisticRegression(examples, η::Float64; steps = typemax(Int))
    N = length(examples)
    M = length(examples[1][1]) + 1
    X = Array{Float64, 2}(undef, (N, M))
    Y = Vector{Float64}(undef, N)
    for (i, (x, y)) in enumerate(examples)
        X[i, :] = [x' 1]
        Y[i] = y
    end
    @show X

    w = rand(M)

    sigmoid(x) = 1 / (1 + exp(-x))
    loss(X, Y, θ) = begin
        P = sigmoid.(X * θ)
        L = @. - Y * log2(P) - (1 - Y) * log2(1 - P)
        sum(L)
    end

    cnt = 0
    loss0 = loss(X, Y, w)
    while true
        A = X * w
        E = sigmoid.(A) - Y
        w -= η * X' * E

        cnt += 1
        loss1 = loss(X, Y, w)
        @show (loss0, loss1)
        if cnt == step || (loss0 - loss1) / loss0 < 0.0001
            break
        end
        loss0 = loss1
    end
    w
end
