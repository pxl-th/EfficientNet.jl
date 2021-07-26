module EfficientNet
export EffNet

using Downloads: download
using Pickle
using Flux

include("params.jl")
include("mb.jl")
include("utils.jl")
include("model.jl")
include("load_utils.jl")

function main()
    device = gpu
    model = EffNet("efficientnet-b0")
    # model = from_pretrained("efficientnet-b0")
    model = model |> testmode! |> device
    @info "Model loaded on device $device."
    @info "Stages channels: $(stages_channels(model))"
    @info "Stages: $(get_stages(model))"

    x = randn(Float32, 256, 256, 3, 1) |> device

    features = model(x, Val(:stages_map))
    features = model(x, Val(:stages_map))
    @info "Features: $(length(features))"
    for f in features
        @info "Feature size: $(size(f))"
    end

    x |> model
end
main()

end
