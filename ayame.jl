using Printf
using LinearAlgebra
using CSV
using StatsBase
import Base.*, Base.+


dn_source = dirname(Base.source_path())
push!(LOAD_PATH, dn_source)
using DL

# 関数の合成
∘(f::Function, g::Function) = x::Tensor-> f(g(x::Tensor))

# Arrayをシャッフルする関数
function shuffle(vx)
    n = length(vx)
    return sample(vx, n, replace=false)
end

function main()
    filename = "iris.csv"
    data = CSV.read(filename)
    n_data, n_type = size(data)
    # n_train = 150 * 0.8 = 120
    n_train::Int64 = n_data * 0.8
    # n_test = 150 - 120  = 30
    n_test::Int64 = n_data - n_train


    # 花の名前 -> Onehot
    # irisデータセットでは3種類の花の名前がある
    花の名前s = unique(data[end])
    # n_hana = 3
    n_hana = length(花の名前s)
    vid2onehot = function(x::Int64)
        onehot = zeros(n_hana, 1)
        onehot[x, 1] = 1.0
        return onehot
    end
    # 花の名前をonehot形式に変換する
    花toOnehot = x::String -> vid2onehot(findfirst(花の名前s .== x))
    # argmax(onehot)を花の名前に変える
    id2name = x -> 花の名前s[x]

    # 入力の次元
    in_features = n_type - 1
    # 出力の次元
    out_features = n_hana

    data = map(i -> [data[i, :]...], 1:n_data)
    # データをshuffle
    data = shuffle(data)
    # 学習データ
    vtrain_x = map(i -> Float64.(data[i][1:in_features]), 1:n_train)
    vtrain_y = map(i -> data[i][n_type], 1:n_train)
    vtrain_y = map(花toOnehot, vtrain_y)
    traindataset = collect(zip(vtrain_x, vtrain_y))
    # テストデータ
    vtest_x = map(i -> Float64.(data[i][1:in_features]), n_train+1:n_data)
    vtest_y = map(i -> data[i][n_type], n_train+1:n_data)
    vtest_y = map(花toOnehot, vtest_y)
    testdataset = collect(zip(vtest_x, vtest_y))


    # 層
    W1 = VariableTensor(randn(32, in_features)/sqrt(32*in_features))
    b1 = VariableTensor(zeros(32, 1))
    W2 = VariableTensor(randn(out_features, 32)/sqrt(out_features*32))
    b2 = VariableTensor(zeros(out_features, 1))
    l1 = x -> W1 * x + b1
    l2 = x -> W2 * x + b2
    # モデル
    model = l2 ∘ ReLU ∘ l1
    # 損失関数
    loss_func = (predict::Tensor, data::Tensor) -> MSE(predict, data)
    # 最適化関数
    optimizer = SGD(1e-4)

    f = open("loss.dat", "w")
    for epoch in 1:1000
        # train
        @show epoch
        train_loss::Float64 = 0
        # traindatasetをシャッフルする
        traindataset = shuffle(traindataset)
        for (batch_idx, (x, y)) in enumerate(traindataset)
            x = ConstantTensor(x)
            y = ConstantTensor(y)
            predict = model(x)
            loss = loss_func(predict, y)
            train_loss += loss.x
            # W1.grad, W2.grad, b1.grad, b2.gradに微分を足しこんでいく
            backward(loss)

            # .gradを使ってW1.x, W2.x, b1.x, b2.xを更新する
            # batch_size = 30で学習する. 
            if batch_idx % 30 == 0 || batch_idx == n_train
                optimizer(loss)
                # .gradを0にする
                zero_grad(loss)
            end
        end

        @show train_loss

        # test
        正解数::Int64 = 0
        for (x, y) in testdataset
            # 入力
            x = ConstantTensor(x)
            # 教師データ
            y = ConstantTensor(y)
            p = model(x)
            pname = id2name(argmax(p.x))
            name = id2name(argmax(y.x))
            正解数 += pname == name
        end
        @printf(f, "%d, % .3e, %.3f\n", epoch, train_loss, 正解数 / n_test)

    end
    close(f)
end


main()