
import CSV
import Random
import Plots

using Statistics

include("k_fold_cross_validation.jl")

"""
    least_squares_regression_closed_solution(X, y)

It takes time `O(M * N * M + M^3 + M * M * N + M * N * 1) = O(M^2 * N + M^3)`
"""
least_squares_regression_closed_solution(X, y) = inv(X' * X) * X' * y

"""
    regularization_l_1(X, y, λ)

It takes time `O(M^2 * N + M^3)`
"""
eye(n) = [i == j ? 1.0 : 0.0 for i = 1:n, j = 1:n]
regularization_l_1(X, y, λ) = begin
    (n, d) = size(X)
    inv(X' * X + λ * eye(d)) * X' * y
end

total_squared_error(X, y, a) = sum((X * a - y).^2)

normalized_error(X, y, a) = sqrt(sum((X * a - y).^2) / sum(y.^2))

function gradient_descent_one_step(X, y, a, λ)
    (n, m) = size(X)
    a - (λ / n) * X' * (X * a - y)
end

"""
    gradient_descent(X, y, a, λ; steps, error)

It takes `O(steps * N * M)`
"""
function gradient_descent(X, y, a, λ; steps = 1000000, error = 1e-5)
    loss_0 = normalized_error(X, y, a)
    for i = 1:steps
        a = gradient_descent_one_step(X, y, a, λ)
        loss_1 = total_squared_error(X, y, a)
        if abs(loss_0 - loss_1) < error
            break
        end
        loss_0 = loss_1
    end
    a
end


function stochastic_gradient_descent_one_step(X, y, a, λ)
    (n, d) = size(X)
    p = rand(1:n)
    a - λ * X[p, :] * (a' * X[p, :] - y[p])
end

"""
    stochastic_gradient_descent(X, y, a, λ; steps, error)

It takes time `O(steps * (N + M))`
"""
function stochastic_gradient_descent(X, y, a, λ; steps = 1000000, error = 1e-5)
    loss_0 = normalized_error(X, y, a)
    for i = 1:steps
        a = stochastic_gradient_descent_one_step(X, y, a, λ)
        loss_1 = total_squared_error(X, y, a)
        if abs(loss_0 - loss_1) < error
            break
        end
        loss_0 = loss_1
    end
    a
end

function generate_data()
    train_n = 100
    test_n = 1000
    d = 100
    X_train = randn((train_n,d))
    a_true = randn((d,1))
    y_train = X_train * a_true + randn((train_n, 1)) / 2
    X_test = randn((test_n,d))
    y_test = X_test * a_true + randn((test_n, 1)) / 2

    X_train, y_train, X_test, y_test, a_true
end

train_data = CSV.read(pwd() * "/data/regression/train_.csv")
test_data = CSV.read(pwd() * "/data/regression/test_.csv")

d = 7
train_n = 80000
test_n  = 30000

function normalize_data!(X)
    normalize(x) = iszero(std(x)) ? x : (x .- mean(x)) ./ std(x)
    (n, d) = size(X)
    for j = 1:d
        X[:, j] = normalize(X[:, j])
    end
    X
end

X_train = begin
    a = zeros(train_n, d)
    for i = 1:train_n
        for j = 1:6
            a[i, j] = train_data[j][i]
        end
        a[i, 7] = 1.0
    end
    a
end |> normalize_data!

y_train = convert(Array{Float64, 1}, train_data[8])


X_test = begin
    a = zeros(test_n, d)
    for i = 1:test_n
        for j = 1:6
            a[i, j] = test_data[j][i]
        end
        a[i, 7] = 1.0
    end
    a
end |> normalize_data!

"""
draw figure for stochastic_gradient_descent
"""
@time begin
    steps = 10000
    (n, d) = size(X_train)
    function draw_pic(λ)
        a = rand(d)
        yy = zeros(steps)
        for i = 1:steps
            a = stochastic_gradient_descent_one_step(X_train, y_train, a, λ)
            yy[i] = normalized_error(X_train, y_train, a)
        end
        yy
    end

    Plots.plot()
    for λ in [0.001, 0.0005, 0.00005]
        yy = draw_pic(λ)
        Plots.plot!(1:steps, yy, label = string(λ))
    end
    Plots.xlabel!("iteration number")
    Plots.ylabel!("normalized_error")
    Plots.title!("stochastic gradient descent")
    # Plots.savefig("stochastic_gradient_descent.png")
end


"""
draw figure for gradient_descent
"""
@time begin
    steps = 10000
    (n, d) = size(X_train)
    function draw_pic(λ)
        a = rand(d)
        yy = zeros(steps)
        for i = 1:steps
            a = gradient_descent_one_step(X_train, y_train, a, λ)
            yy[i] = normalized_error(X_train, y_train, a)
        end
        yy
    end

    Plots.plot()
    for λ in [0.001, 0.0005, 0.00005]
        yy = draw_pic(λ)
        Plots.plot!(1:steps, yy, label = string(λ))
    end
    Plots.xlabel!("iteration number")
    Plots.ylabel!("normalized_error")
    Plots.title!("gradient descent")
    # Plots.savefig("gradient_descent.png")
end

"""
gradient descent
"""
@time begin
    steps = [0.005, 0.001, 0.0005]
    map(steps) do λ
        @show λ
        train_model(X, y) = begin
            (n, d) = size(X)
            gradient_descent(X, y, zeros(d), λ, steps = 8000)
        end
        f(a, X, y) = normalized_error(X, y, a)
        k_fold_cross_validation(4, X_train, y_train, train_model, f)
    end
end

"""
stochastic_gradient_descent
"""
@time begin
    #steps = [0.005, 0.001, 0.0005]
    steps = [0.001]
    map(steps) do λ
        @show λ
        train_model(X, y) = begin
            (n, d) = size(X)
            stochastic_gradient_descent(X, y, zeros(d), λ, steps = 10000)
        end
        f(a, X, y) = normalized_error(X, y, a)
        k_fold_cross_validation(4, X_train, y_train, train_model, f)
    end
end

"""
v5.0 stochastic_gradient_descent λ = 0.0005
"""
@time begin
    (n, d) = size(X_train)
    a = stochastic_gradient_descent(X_train, y_train, zeros(d), 0.0005, steps = 10000)
    y_test = X_test * a
    open("regression_v5.txt", "w") do f
        n = length(y_test)
        for i = 1:n
            println(f, y_test[i])
        end
    end
end

"""
v6.0 gradient_descent λ = 0.001
"""
@time begin
    (n, d) = size(X_train)
    a = gradient_descent(X_train, y_train, zeros(d), 0.001, steps = 5000)
    y_test = X_test * a
    open("regression_v6.txt", "w") do f
        n = length(y_test)
        for i = 1:n
            println(f, y_test[i])
        end
    end
end

@time begin
        train_model(X, y) = begin
            least_squares_regression_closed_solution(X, y)
        end
        f(a, X, y) = normalized_error(X, y, a)
        k_fold_cross_validation(4, X_train, y_train, train_model, f)
end

"""
v7.0 least_squares_regression_closed_solution
"""
@time begin
    a = least_squares_regression_closed_solution(X_train, y_train)
    y_test = X_test * a
    open("regression_v7.txt", "w") do f
        n = length(y_test)
        for i = 1:n
            println(f, y_test[i])
        end
    end
end

"""
regularization_l_1
"""
@time begin
    λs = [0.0005, 0.005, 0.05, 0.5, 5, 50, 500]
    map(λs) do λ
        train_model(X, y) = begin
            regularization_l_1(X, y, λ)
        end
        f(a, X, y) = normalized_error(X, y, a)
        k_fold_cross_validation(4, X_train, y_train, train_model, f)
    end
end

"""
v8.0 regularization_l_1 λ = 0.5
"""
@time begin
    a = regularization_l_1(X_train, y_train, 0.5)
    y_test = X_test * a
    open("regression_v8.txt", "w") do f
        n = length(y_test)
        for i = 1:n
            println(f, y_test[i])
        end
    end
end


####################################################
#
#       quadratic regression
#
####################################################
train_data = CSV.read(pwd() * "/data/regression/train_.csv")
test_data = CSV.read(pwd() * "/data/regression/test_.csv")

d = 28
train_n = 80000
test_n  = 30000

X_train = begin
    a = zeros(train_n, 28)
    for i = 1:train_n
        for j = 1:6
            a[i, j] = train_data[j][i]
        end
        a[i, 7] = 1.0
        dim = 7
        for j = 1:6
            for k = j:6
                dim += 1
                a[i, dim] = a[i, j] * a[i, k]
            end
        end
    end
    normalize_data!(a)
end

y_train = convert(Array{Float64, 1}, train_data[8])

X_test = begin
    a = zeros(test_n, 28)
    for i = 1:test_n
        for j = 1:6
            a[i, j] = test_data[j][i]
        end
        a[i, 7] = 1.0
        dim = 7
        for j = 1:6
            for k = j:6
                dim += 1
                a[i, dim] = a[i, j] * a[i, k]
            end
        end
    end
    normalize_data!(a)
end

"""
v9.0 least_squares_regression_closed_solution
"""
@time begin
    a = least_squares_regression_closed_solution(X_train, y_train)
    y_test = X_test * a
    open("regression_v9.txt", "w") do f
        n = length(y_test)
        for i = 1:n
            println(f, y_test[i])
        end
    end
end


"""
v10.0 regularization_l_1 λ = 5
"""
@time begin
    a = regularization_l_1(X_train, y_train, 5)
    y_test = X_test * a
    open("regression_v10.txt", "w") do f
        n = length(y_test)
        for i = 1:n
            println(f, y_test[i])
        end
    end
end

####################################################
#
#       cubic regression
#
####################################################

sum(1 for i = 1:6 for j = i:6)

sum(1 for i = 1:6 for j = i:6 for k = j:6)

1 + 6 + 21 + 56

d = 84
train_n = 80000
test_n  = 30000

X_train = begin
    a = zeros(train_n, d)
    for i = 1:train_n
        a[i, 1] = 1.0
        dim = 1
        for j = 1:6
            dim += 1
            a[i, dim] = train_data[j][i]
        end
        for j = 1:6
            for k = j:6
                dim += 1
                a[i, dim] = a[i, j + 1] * a[i, k + 1]
            end
        end
        for j = 1:6
            for k = j:6
                for t = k:6
                    dim += 1
                    a[i, dim] = a[i, j + 1] * a[i, k + 1] * a[t + 1]
                end
            end
        end
    end
    normalize_data!(a)
end

y_train = convert(Array{Float64, 1}, train_data[8])

X_test = begin
    a = zeros(test_n, d)
    for i = 1:test_n
        a[i, 1] = 1.0
        dim = 1
        for j = 1:6
            dim += 1
            a[i, dim] = test_data[j][i]
        end
        for j = 1:6
            for k = j:6
                dim += 1
                a[i, dim] = a[i, j + 1] * a[i, k + 1]
            end
        end
        for j = 1:6
            for k = j:6
                for t = k:6
                    dim += 1
                    a[i, dim] = a[i, j + 1] * a[i, k + 1] * a[t + 1]
                end
            end
        end
    end
    normalize_data!(a)
end

"""
v11.0 regularization_l_1 λ = 5
"""
@time begin
    a = regularization_l_1(X_train, y_train, 5)
    y_test = X_test * a
    open("regression_v11.txt", "w") do f
        n = length(y_test)
        for i = 1:n
            println(f, y_test[i])
        end
    end
end
