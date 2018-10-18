"""
    LinearRegression
"""
module LinearRegression

export least_squares_regression_closed_solution
export regularization_l_1
export total_squared_error, normalized_error
export gradient_descent_one_step, gradient_descent
export stochastic_gradient_descent_one_step, stochastic_gradient_descent

using Statistics

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

end
