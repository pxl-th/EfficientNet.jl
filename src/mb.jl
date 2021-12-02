mutable struct MBConv{E, D, X, P, S}
    expansion::E
    depthwise::D
    excitation::X
    projection::P
    dropout::S

    do_expansion::Bool
    do_excitation::Bool
    do_skip::Bool

    active::Union{Bool, Nothing}
end
Flux.@functor MBConv

"""
Mobile Inverted Residual Bottleneck Block.

Args:
    in_channels: Number of input channels.
    out_channels: Number of output channels.
    expansion_ratio:
        Expansion ratio defines the number of output channels.
        Set to `1` to disable expansion phase.
        `out_channels = input_channels * expansion_ratio`.
    kernel: Size of the kernel for the depthwise conv phase.
    stride: Size of the stride for the depthwise conv phase.
    momentum: BatchNorm momentum.
    ϵ: BatchNorm ϵ.
    se_ratio:
        Squeeze-Excitation ratio. Should be in `(0, 1]` range.
        Set to `-1` to disable.
    skip_connection: Whether to use skip connection and drop connect.
"""
function MBConv(
    in_channels, out_channels, kernel, stride;
    expansion_ratio, se_ratio, skip_connection, momentum, ϵ,
)
    do_skip = skip_connection && stride == 1 && in_channels == out_channels
    do_expansion, do_excitation = expansion_ratio != 1, 0 < se_ratio ≤ 1
    drop, pad, bias = Dropout(0.0; dims=3), SamePad(), false

    mid_channels = ceil(Int, in_channels * expansion_ratio)
    excitation = nothing
    expansion = do_expansion ? VChain(Conv((1, 1), in_channels=>mid_channels; bias, pad), BatchNorm(mid_channels, swish; momentum, ϵ)) : nothing
    depthwise = VChain(Conv(kernel, mid_channels=>mid_channels; bias, stride, pad, groups=mid_channels), BatchNorm(mid_channels, swish; momentum, ϵ))
    if do_excitation
        n_squeezed_channels = max(1, ceil(Int, in_channels * se_ratio))
        excitation = VChain(
            AdaptiveMeanPool((1, 1)),
            Conv((1, 1), mid_channels=>n_squeezed_channels, swish; pad),
            Conv((1, 1), n_squeezed_channels=>mid_channels; pad))
    end
    projection = VChain(Conv((1, 1), mid_channels=>out_channels; pad, bias), BatchNorm(out_channels; momentum, ϵ))
    MBConv(expansion, depthwise, excitation, projection, drop, do_expansion, do_excitation, do_skip, nothing)
end

function Flux.testmode!(m::MBConv, mode = true)
    m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode
    m
end

function (m::MBConv)(x; drop_probability::Union{Real, Nothing} = nothing)
    if m.do_expansion
        o = x |> m.expansion |> m.depthwise
    else
        o = x |> m.depthwise
    end
    if m.do_excitation
        o = σ.(o |> m.excitation) .* o
    end
    o = o |> m.projection
    if m.do_skip
        # The combination of skip connection and drop connect creates stochastic depth effect.
        if drop_probability ≢ nothing
            m.dropout.p = drop_probability
            o = o |> m.dropout
        end
        o = o + x
    end
    o
end

function Base.show(io::IO, m::MBConv{E, D, X, P, S}) where {E, D, X, P, S}
    print(io, "MBConv:\n",
        "- expansion: ", E, "\n",
        "- depthwise: ", D, "\n",
        "- excitation: ", X, "\n",
        "- projection: ", P, "\n",
        "- dropout: ", S, "\n",
        "- do expansion: ", m.do_expansion, "\n",
        "- do excitation: ", m.do_excitation, "\n",
        "- do skip: ", m.do_skip, "\n",
        "- active: ", m.active, "\n")
end
