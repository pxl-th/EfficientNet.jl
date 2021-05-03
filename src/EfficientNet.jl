module EfficientNet
export EffNet, from_pretrained

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

function from_pretrained(
    model_name::String, cache_dir::Union{String, Nothing} = nothing,
)
    url_base = (
        "https://github.com/lukemelas/EfficientNet-PyTorch" *
        "/releases/download/1.0/"
    )
    url_map = Dict(
        "efficientnet-b0" => "efficientnet-b0-355c32eb.pth",
        "efficientnet-b1" => "efficientnet-b1-f1951068.pth",
        "efficientnet-b2" => "efficientnet-b2-8bb594d6.pth",
        "efficientnet-b3" => "efficientnet-b3-5fb5a3c3.pth",
        "efficientnet-b4" => "efficientnet-b4-6ed6700e.pth",
        "efficientnet-b5" => "efficientnet-b5-b6417697.pth",
        "efficientnet-b6" => "efficientnet-b6-c76e70fd.pth",
        "efficientnet-b7" => "efficientnet-b7-dcc49843.pth",
    )
    if !(model_name in keys(url_map))
        error(
            "Invalid model name: $model_name. " *
            "Supported pretrained models: $(keys(url_map)) "
        )
    end

    if cache_dir ≡ nothing
        cache_dir = joinpath(homedir(), ".cache", "EfficientNet.jl")
        !isdir(cache_dir) && mkdir(cache_dir)
        @info "Using default cache dir $cache_dir"
    end

    params_file = url_map[model_name]
    params_path = joinpath(cache_dir, params_file)
    if !isfile(params_path)
        download_url = url_base * params_file
        @info(
            "Downloading $model_name params:\n" *
            "\t- from URL: $download_url \n" *
            "\t- to directory: $params_path"
        )
        download(download_url, params_path)
        @info "Finished downloading params."
    end

    block_params, global_params = get_model_params(model_name)
    model = EffNet(block_params, global_params)

    params = Pickle.Torch.THload(params_path)
    _load_pth!(model, params)

    model
end

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

main()
test_mbconv()
test_train()

end
