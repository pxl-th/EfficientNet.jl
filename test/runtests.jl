using Test
using Flux
using EfficientNet

@testset "Test MBConv forward/backward passes" begin
    device = gpu

    m = EfficientNet.MBConv(
        3, 3, (3, 3), 1,
        expansion_ratio=2f0, se_ratio=0.5f0, skip_connection=true,
        momentum=0.99f0, ϵ=1f-6,
    )
    m = m |> trainmode! |> device

    trainables = m |> params
    @test length(trainables) == 13

    x = randn(Float32, 32, 32, 3, 1) |> device
    y = randn(Float32, 32, 32, 3, 1) |> device

    o = m(x; drop_probability=0.2f0)
    @test size(o) == size(y)

    gs = gradient(trainables) do
        o = m(x; drop_probability=0.2f0)
        Flux.mse(o, y)
    end
end

@testset "Test EffNet forward/backward passes" begin
    device = gpu

    model = EffNet("efficientnet-b0"; n_classes=10)
    model = model |> trainmode! |> device
    trainables = model |> params

    x = randn(Float32, 224, 224, 3, 1) |> device
    y = randn(Float32, 10, 1) |> device

    x |> model
    endpoints = extract(model, x)
    @test length(endpoints) == 6

    gs = gradient(trainables) do
        o = x |> model |> softmax
        Flux.crossentropy(o, y)
    end
end
