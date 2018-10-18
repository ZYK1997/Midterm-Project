"""
    RegressionDecisionTree
"""
module RegressionDecisionTree

export getLabel, build, find_best, find_proper, find_possible

using Statistics
using Random

struct Leaf
    label::Float64
end

struct Node
    attribute::Int
	value::Float64
    son::Tuple{Union{Leaf, Node}, Union{Leaf, Node}}
end

getLabel(t::Leaf, v) = t.label
getLabel(t::Node, v) = getLabel(t.son[v[t.attribute] < t.value ? 1 : 2], v)

function splitData(X, y, d, v)
	I = X[:, d] .< v
	X[I, :], y[I], X[xor.(I, true), :], y[xor.(I, true)]
end

function splitData_only_y(X, y, d, v)
	I = X[:, d] .< v
	y[I], y[xor.(I, true)]
end

function build(
    X,
	y,
    findAttribute::Function;
	E::Float64 = 1.0,
	least_size::Int = 4
	)::Union{Leaf, Node}

    d, val = findAttribute(X, y, E, least_size)
	if isequal(d, nothing)
		return Leaf(val)
	end

    X1, y1, X2, y2 = splitData(X, y, d, val)
    sons = (build(X1, y1, findAttribute, E = E, least_size = least_size),
			build(X2, y2, findAttribute, E = E, least_size = least_size))
    Node(d, val, sons)
end

"""
	find_best(X, y, attributes)

attr <- min_d min_v (n_1 σ_1^2 + n_2 σ_2^2)
"""
function find_best(X, y, E, least_size)
	cost(y) = sum(abs2, y .- mean(y))
	function cost(d::Int, v::Float64)
		y1, y2 = splitData_only_y(X, y, d, v)
		cost(y1) + cost(y2)
	end

	(n, m) = size(X)
	c0 = cost(y)
	Best_d, Best_v, Best_c = 0, 0, Inf
	for d = 1:m
		vs = X[:, d]
		sort!(vs)
		best_v, best_c = 0, Inf
		for i = 1:length(vs) - 1
			v = (vs[i] + vs[i + 1]) / 2
			c = cost(d, v)
			if c < best_c
				best_v, best_c = v, c
			end
		end
		if best_c < Best_c
			Best_d, Best_v, Best_c = d, best_v, best_c
		end
	end
	if c0 - Best_c < E
		return nothing, mean(y)
	end
	y1, y2 = splitData_only_y(X, y, Best_d, Best_v)
	if length(y1) < least_size || length(y2) < least_size
		return nothing, mean(y)
	end
	Best_d, Best_v
end

"""
	find_possible(X, y, attributes)

attr <- min_d min_v (n_1 σ_1^2 + n_2 σ_2^2)
"""
function find_possible(X, y, E::Float64, least_size::Int)
	cost(y) = sum(abs2, y .- mean(y))
	function cost(d::Int, v::Float64)
		y1, y2 = splitData_only_y(X, y, d, v)
		cost(y1) + cost(y2)
	end

	limit = 100
	(n, m) = size(X)
	c0 = cost(y)
	Best_d, Best_v, Best_c = 0, 0, Inf
	for d = 1:m
		vs = X[:, d]
		sort!(vs)
		best_v, best_c = 0, Inf
		perm = shuffle(1:n - 1)[1:min(limit, n - 1)]
		for p in perm
			v = (vs[p] + vs[p + 1]) / 2
			c = cost(d, v)
			if c < best_c
				best_v, best_c = v, c
			end
		end
		if best_c < Best_c
			Best_d, Best_v, Best_c = d, best_v, best_c
		end
	end
	if c0 - Best_c < E
		return nothing, mean(y)
	end
	y1, y2 = splitData_only_y(X, y, Best_d, Best_v)
	if length(y1) < least_size || length(y2) < least_size
		return nothing, mean(y)
	end
	Best_d, Best_v
end

function find_proper(X, y, E, least_size)
	(n, d) = size(X)
	limit = 100
	f = n < limit ? find_best : find_possible
	f(X, y, E, least_size)
end

end
