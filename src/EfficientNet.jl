module EfficientNet
export EffNet, from_pretrained, stages_channels

using Downloads: download
using Pickle
using Flux

include("params.jl")
include("mb.jl")
include("utils.jl")
include("model.jl")
include("load_utils.jl")

end
