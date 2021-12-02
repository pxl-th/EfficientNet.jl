module EfficientNet
export EffNet

using Downloads: download
using Pickle
using Flux

applychain(fs, x) = isempty(fs) ? x : applychain(fs[2:end], first(fs)(x))

struct VChain
    layers
    VChain(xs...) = new(collect(xs))
end
Flux.@forward VChain.layers Base.getindex, Base.length, Base.first, Base.last,
    Base.iterate, Base.lastindex, Base.keys
Flux.functor(::Type{<:VChain}, c) = c.layers, ls -> VChain(ls...)
(c::VChain)(x) = applychain(c.layers, x)

include("params.jl")
include("mb.jl")
include("model.jl")
include("load_utils.jl")

end
