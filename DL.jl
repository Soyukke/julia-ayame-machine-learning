module DL

using Printf
using LinearAlgebra
import Base.*, Base.+

export (*), (+), ReLU, VariableTensor, MSE, ConstantTensor, backward, zero_grad, SGD, Linear, Tensor

# Tensor
abstract type Tensor end
abstract type LeafTensor <: Tensor end
abstract type FunctionalTensor <: Tensor end


#= 掛け算 t = t1 * t2 =#
mutable struct MulTensor <: FunctionalTensor
    x
    tensors::Array{Tensor, 1}
end
(*)(t3::Tensor, t4::Tensor) = abstract_operator(MulTensor, (x1, x2) -> x1 * x2, [t3, t4])
# t2 <- t1 <- t3 * t4
function grad_fn(t1::MulTensor, grad_up)
    t3, t4 = t1.tensors
    # dt2/dt1 * dt1/dt3 = grad_up * t4.x
    # dt2/dt1 * dt1/dt4 = grad_up * t3.x
    return [t4.x * grad_up, grad_up * t3.x]
end


#= 足し算 =#
mutable struct AddTensor <: FunctionalTensor
    x
    tensors::Array{Tensor, 1}
end
# forward
(+)(t3::Tensor, t4::Tensor) = abstract_operator(AddTensor, (x1, x2) -> x1 .+ x2, [t3, t4])
# backward t2 <- t1 <- t3 + t4
function grad_fn(t1::AddTensor, grad_up)
    t3, t4 = t1.tensors
    # dt2/dt1 * dt1/dt3 = grad_up * I
    # dt2/dt1 * dt1/dt4 = grad_up * I
    return [grad_up, grad_up]
end


#= ReLU function t = ReLU(t1) =#
mutable struct ReLUTensor <: FunctionalTensor
    x
    tensors::Array{Tensor, 1}
end
# forward
function ReLU(t::Tensor)
    return abstract_operator(ReLUTensor, x -> x .* (x .> 0), [t])
end
# backward t2 <- t1 <- ReLU(t3)
function grad_fn(t1::ReLUTensor, grad_up)
    t3 = t1.tensors[1]
    # dt2/dt1 * dt1/dt3 = grad_up * d ReLU(x)/dx
    return [grad_up .* (t3.x' .> 0)]
end


#= Mean Squared Error t = MSE(t1, t2) =#
mutable struct MSETensor <: FunctionalTensor
    x
    tensors::Array{Tensor, 1}
end
# forward
function MSE(t1::Tensor, t2::Tensor)
    mse = (x1, x2) -> sum((x1 - x2).^2) / length(x1)
    return abstract_operator(MSETensor, mse, [t1, t2])
end
# t2 <- t1 <- MSE(t3, t4)
function grad_fn(t1::MSETensor, grad_up)
    t3, t4 = t1.tensors
    # dt2/dt1 * dt1/dt3 = grad_up * d/dx sum((x - y)^2) / length(x)
    N = length(t3.x)
    dt1dt3 = 2*(t3.x - t4.x)' / N
    dt1dt4 = 2*(t4.x - t3.x)' / N
    return [grad_up * dt1dt3, grad_up * dt1dt4]
end


#= t2 <- t1 末端の部分 微分あり =#
mutable struct VariableTensor <: LeafTensor
    x
    grad
    VariableTensor(x) = new(x, zero(x))
end

#= ただの定数 =#
mutable struct ConstantTensor <: LeafTensor
    x
end


# backward 計算
# 最初
function backward(t::Tensor)
	backward(t, 1)
end
# 再帰
function backward(t::Tensor, grad_up)
    # 微分はtが持つTensorsの数だけあるから，リストで返すことにする．
    vgrad = grad_fn(t, grad_up)
	# 微分と対応するTensorのペアをつくる
    pair = zip(t.tensors, vgrad)
    # 掘り下げ
	map(x -> backward(x[1], x[2]), pair)
end
# ConstantTensorとか
function backward(t::LeafTensor, grad_up)
end
# VariableTensorは勾配を蓄積していく
function backward(t::VariableTensor, grad_up)
    t.grad += grad_up'
end


# 抽象化された関数
# 演算子を計算するときにConstant同士の演算の出力はConstantであるようにする
# Array{T, 1} where T <: Tensor という書き方でTensor以下のタイプのTのArrayを網羅できる
function abstract_operator(EnzansiTensor::DataType, f::Function, tensors::Array{T, 1} where T <: Tensor)
    # Tensor.xを展開してfに代入する
    y = f(map(tensor -> tensor.x, tensors)...)
    return EnzansiTensor(y, tensors)
end
function abstract_operator(EnzansiTensor::DataType, f::Function, tensors::Array{ConstantTensor, 1})
    y = f(map(tensor -> tensor.x, tensors)...)
    return ConstantTensor(y)
end

function tensor_map(f::Function, t::Tensor)
    #= 任意の関数をVariableTensorに作用させる
    毎回掘り下げる必要はないと思うので考えなおす余地あり
    =#
    # 掘り下げていく
    map(t -> tensor_map(f, t), t.tensors)
end
function tensor_map(f::Function, t::LeafTensor)
    # 末端だけどVariableTensor以外のtypeはパラメータを変えない
    nothing
end
function tensor_map(f::Function, t::VariableTensor)
    # VariableTensorにだけ作用させる 
    f(t)
end


function zero_grad(tensor::Tensor)
    f = t -> t.grad = zero(t.grad)
    tensor_map(f, tensor)
end


function SGD(lr=1e-3)
    f = t::Tensor -> t.x = t.x - lr * t.grad
    optimizer = t::Tensor -> tensor_map(f, t)
    return optimizer
end


function Linear(in_features::Int64, out_features::Int64)
    W = VariableTensor(randn(out_features, in_features))
    b = VariableTensor(zeros(out_features, 1))
    @show b.x
    return x -> W*x + b
end


function printarray(x)
    show(stdout, "text/plain", x)
    @printf "\n"
end


end
