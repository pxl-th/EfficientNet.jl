struct MBConv{E, D, X, P}
    expansion::E
    depthwise::D
    excitation::X
    projection::P

    do_expansion::Bool
    do_excitation::Bool
    do_skip::Bool
end
Flux.@functor MBConv

struct FusedMBConv{E, X, P}
    expansion::E
    excitation::X
    projection::P

    do_expansion::Bool
    do_excitation::Bool
    do_skip::Bool
end
Flux.@functor FusedMBConv

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
    se_ratio:
        Squeeze-Excitation ratio. Should be in `(0, 1]` range.
        Set to `-1` to disable.
    skip_connection: Whether to use skip connection and drop connect.
"""
function MBConv(
    in_channels, out_channels, kernel, stride;
    expansion_ratio, se_ratio, skip_connection,
)
    do_skip = skip_connection && stride == 1 && in_channels == out_channels
    do_expansion, do_excitation = expansion_ratio != 1, 0 < se_ratio ≤ 1
    pad, bias = SamePad(), false

    mid_channels = ceil(Int, in_channels * expansion_ratio)
    expansion = do_expansion ?
        Chain(
            Conv((1, 1), in_channels=>mid_channels; bias, pad),
            BatchNorm(mid_channels, swish)) :
        nothing

    depthwise = Chain(
        Conv(kernel, mid_channels=>mid_channels; bias, stride, pad, groups=mid_channels),
        BatchNorm(mid_channels, swish))

    excitation = nothing
    if do_excitation
        n_squeezed_channels = max(1, ceil(Int, in_channels * se_ratio))
        excitation = Chain(
            AdaptiveMeanPool((1, 1)),
            Conv((1, 1), mid_channels=>n_squeezed_channels, swish; pad),
            Conv((1, 1), n_squeezed_channels=>mid_channels; pad))
    end
    projection = Chain(
        Conv((1, 1), mid_channels=>out_channels; pad, bias),
        BatchNorm(out_channels))
    MBConv(
        expansion, depthwise, excitation, projection, do_expansion,
        do_excitation, do_skip)
end

function (m::MBConv)(x)
    if m.do_expansion
        o = m.depthwise(m.expansion(x))
    else
        o = m.depthwise(x)
    end
    if m.do_excitation
        o = σ.(m.excitation(o)) .* o
    end
    o = m.projection(o)
    if m.do_skip
        o = o + x
    end
    o
end

function Base.show(io::IO, m::FusedMBConv{E, X, P}) where {E, X, P}
    print(io, "FusedMBConv:\n",
        "- expansion: ", E, "\n",
        "- excitation: ", X, "\n",
        "- projection: ", P, "\n",
        "- do expansion: ", m.do_expansion, "\n",
        "- do excitation: ", m.do_excitation, "\n",
        "- do skip: ", m.do_skip, "\n")
end

"""
Fused Mobile Inverted Residual Bottleneck Block.

Args:
    in_channels: Number of input channels.
    out_channels: Number of output channels.
    expansion_ratio:
        Expansion ratio defines the number of output channels.
        Set to `1` to disable expansion phase.
        `out_channels = input_channels * expansion_ratio`.
    kernel: Size of the kernel for the depthwise conv phase.
    stride: Size of the stride for the depthwise conv phase.
    se_ratio:
        Squeeze-Excitation ratio. Should be in `(0, 1]` range.
        Set to `-1` to disable.
    skip_connection: Whether to use skip connection and drop connect.
"""
function FusedMBConv(
    in_channels, out_channels, kernel, stride;
    expansion_ratio, se_ratio, skip_connection,
)
    do_skip = skip_connection && stride == 1 && in_channels == out_channels
    do_expansion, do_excitation = expansion_ratio != 1, 0 < se_ratio ≤ 1
    pad, bias = SamePad(), false

    mid_channels = ceil(Int, in_channels * expansion_ratio)
    expansion = do_expansion ?
        Chain(
            Conv((3, 3), in_channels=>mid_channels; bias, pad),
            BatchNorm(mid_channels, swish)) :
        nothing

    excitation = nothing
    if do_excitation
        n_squeezed_channels = max(1, ceil(Int, in_channels * se_ratio))
        excitation = Chain(
            AdaptiveMeanPool((1, 1)),
            Conv((1, 1), mid_channels=>n_squeezed_channels, swish; pad),
            Conv((1, 1), n_squeezed_channels=>mid_channels; pad))
    end
    projection = Chain(
        Conv((1, 1), mid_channels=>out_channels; pad, bias),
        BatchNorm(out_channels))
    FusedMBConv(
        expansion, excitation, projection, do_expansion,
        do_excitation, do_skip)
end

function (m::FusedMBConv)(x)
    if m.do_expansion
        o = m.expansion(x)
    else
        o = x
    end
    if m.do_excitation
        o = σ.(m.excitation(o)) .* o
    end
    o = m.projection(o)
    if m.do_skip
        o = o + x
    end
    o
end

function Base.show(io::IO, m::MBConv{E, D, X, P}) where {E, D, X, P}
    print(io, "MBConv:\n",
        "- expansion: ", E, "\n",
        "- depthwise: ", D, "\n",
        "- excitation: ", X, "\n",
        "- projection: ", P, "\n",
        "- do expansion: ", m.do_expansion, "\n",
        "- do excitation: ", m.do_excitation, "\n",
        "- do skip: ", m.do_skip, "\n")
end
