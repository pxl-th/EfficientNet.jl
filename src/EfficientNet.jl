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

function main()
    device = gpu
    model = from_pretrained("efficientnet-b3")
    model = model |> testmode! |> device
    @info "Model loaded on device $device."

    images = [
        "/home/pxl-th/Downloads/elephant.png",
        "/home/pxl-th/Downloads/elephant2-r.jpg",
        "/home/pxl-th/Downloads/bee.png",
        "/home/pxl-th/Downloads/dog.png",
        "/home/pxl-th/Downloads/spaceshuttle-r.jpg",
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
main()

end
