"""
    Boosting
"""
module Boosting

export boosting_regression, predict

function boosting_regression(model_train, model_predict, X, y; model_num = 30)
    model_list = []
    residuals = y
    for i = 1:model_num
        println("train model ", i)
        model = model_train(X, residuals)
        push!(model_list, model)
        predict = model_predict(model, X)
        residuals -= predict
    end
    model_list
end

function predict(model_list, X, model_predict)
    (n, d) = size(X)
    predict = zeros(n)
    for model in model_list
        predict += model_predict(model, X)
    end
    predict
end

end
