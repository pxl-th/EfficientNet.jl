module EfficientNet
export EffNet, from_pretrained

using Downloads: download
using Pickle
using Flux

include("params.jl")
include("mb.jl")
include("utils.jl")
include("model.jl")
include("load_utils.jl")

# using FileIO
# using Images
# function main()
#     device = gpu
#     model = from_pretrained("efficientnet-b0")
#     model = model |> testmode! |> device
#     @info "Model loaded on device $device."

#     images = [
#         "/home/pxl-th/Downloads/elephant.png",
#         # "/home/pxl-th/Downloads/elephant2-r.jpg",
#         # "/home/pxl-th/Downloads/bee.png",
#         # "/home/pxl-th/Downloads/dog.png",
#         # "/home/pxl-th/Downloads/spaceshuttle-r.jpg",
#     ]
#     for image in images
#         x = Images.load(image) |> channelview .|> Float32
#         x .-= reshape([0.485, 0.456, 0.406], (3, 1, 1))
#         x ./= reshape([0.229, 0.224, 0.225], (3, 1, 1))
#         x = Flux.unsqueeze(permutedims(x, (3, 2, 1)), 4)
#         x = x |> device
#         @info "Image size: $(size(x))"

#         features = model(x, Val(:stages))
#         @info "Features: $(length(features))"
#         for f in features
#             @info "Feature size: $(size(f))"
#         end

#         @info "Image $image:"
#         o = x |> model |> softmax |> cpu
#         o = sortperm(o[:, 1])
#         @info "Top 5 classes: $(o[end:-1:end - 5] .- 1)"
#     end
# end
# main()

end
