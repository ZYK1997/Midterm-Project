
function k_fold_cross_validation(K, X, y, train_model, f)
    (n, m) = size(X)
    len = floor(Int, n / K)
    map(1:K) do i
        test_X = X[(i - 1) * len + 1 : i * len, :]
        test_y = y[(i - 1) * len + 1 : i * len, :]
        train_X = Array{Float64, 2}(undef, n - len, m)
        train_y = Vector{Float64}(undef, n - len)
        train_X[1 : (i - 1) * len, :] = X[1 : (i - 1) * len, :]
        train_y[1 : (i - 1) * len, :] = y[1 : (i - 1) * len, :]
        train_X[(i - 1) * len + 1 : end, :] = X[i * len + 1 : end, :]
        train_y[(i - 1) * len + 1 : end, :] = y[i * len + 1 : end, :]

        model = train_model(train_X, train_y)
        f(model, test_X, test_y)
    end |> sum |> (x -> x / K)
end
