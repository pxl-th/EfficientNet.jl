module EfficientNet
export EffNet, from_pretrained, extract

using Downloads: download
using FileIO
using Images
using Pickle

using CUDA
using Flux

include("params.jl")
include("mb.jl")
include("utils.jl")
include("model.jl")
include("load_utils.jl")

"""
TODO:
- Move BP/FP to tests
- Train on MNIST
- Write README
"""

function main()
    device = gpu
    model = from_pretrained("efficientnet-b3")
    model = model |> testmode! |> device
    @info "Model loaded."

    images = [
        raw"C:\Users\tonys\Downloads\elephant.png",
        raw"C:\Users\tonys\Downloads\elephant2.png",
        raw"C:\Users\tonys\Downloads\bee.png",
        raw"C:\Users\tonys\Downloads\dog.png",
        raw"C:\Users\tonys\Downloads\pug.png",
        raw"C:\Users\tonys\Downloads\spaceshuttle.png",
    ]
    for image in images
        x = Images.load(image) |> channelview .|> Float32
        x .-= reshape([0.485, 0.456, 0.406], (3, 1, 1))
        x ./= reshape([0.229, 0.224, 0.225], (3, 1, 1))
        x = Flux.unsqueeze(permutedims(x, (3, 2, 1)), 4)

        @info "Image $image:"
        o = x |> device |> model |> softmax |> cpu
        o = sortperm(o[:, 1])
        @info "Top 5 classes: $(o[end:-1:end - 5] .- 1)"
    end
end

function test_train()
    device = gpu

    model = from_pretrained("efficientnet-b0")
    model = model |> trainmode! |> device
    @info "Model loaded."
    trainables = model |> params

    x = randn(Float32, 224, 224, 3, 1) |> device
    y = randn(Float32, 1000, 1) |> device

    @info "Forward pass..."
    x |> model
    @info "Feature extraction pass..."
    endpoints = extract(model, x)
    @info typeof(endpoints), length(endpoints)

    @info "Gradient pass..."
    gs = gradient(trainables) do
        o = x |> model |> softmax
        Flux.crossentropy(o, y)
    end

    @info gs
end

function test_mbconv()
    device = gpu

    m = MBConv(
        3, 3, (3, 3), 1,
        expansion_ratio=2f0, se_ratio=0.5f0, skip_connection=true,
        momentum=0.99f0, ϵ=1f-6,
    )
    m = m |> trainmode! |> device
    @show m

    trainables = m |> params
    @info length(trainables)
    @info size.(trainables)

    x = randn(Float32, 32, 32, 3, 1) |> device
    y = randn(Float32, 32, 32, 3, 1) |> device

    @info "Forward pass..."
    m(x; drop_probability=0.2f0)

    @info "Gradient pass..."
    gs = gradient(trainables) do
        o = m(x; drop_probability=0.2f0)
        Flux.mse(o, y)
    end

    @info gs
end

# main()
# test_mbconv()
test_train()

end
