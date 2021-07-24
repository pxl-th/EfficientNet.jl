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
    expansion_ratio, se_ratio, skip_connection,
    momentum, ϵ,
)
    do_expansion = expansion_ratio != 1
    do_excitation = 0 < se_ratio ≤ 1
    do_skip = skip_connection && stride == 1 && in_channels == out_channels
    drop = Dropout(0f0; dims=3)

    mid_channels = ceil(Int, in_channels * expansion_ratio)
    expansion, excitation = nothing, nothing

    # Expansion phase.
    if do_expansion
        expand_conv = Conv(
            (1, 1), in_channels=>mid_channels, bias=false, pad=SamePad(),
        )
        bn0 = BatchNorm(mid_channels, swish; momentum, ϵ)
        expansion = Chain(expand_conv, bn0)
    end

    # Depthwise phase.
    depthwise_conv = Conv(
        kernel, mid_channels=>mid_channels,
        bias=false, stride=stride, pad=SamePad(), groups=mid_channels,
    )
    bn1 = BatchNorm(mid_channels, swish; momentum, ϵ)
    depthwise = Chain(depthwise_conv, bn1)

    # Squeeze and Excitation phase.
    if do_excitation
        n_squeezed_channels = max(1, ceil(Int, in_channels * se_ratio))
        squeeze_conv = Conv(
            (1, 1), mid_channels=>n_squeezed_channels, swish; pad=SamePad(),
        )
        excite_conv = Conv(
            (1, 1), n_squeezed_channels=>mid_channels; pad=SamePad(),
        )
        excitation = Chain(AdaptiveMeanPool((1, 1)), squeeze_conv, excite_conv)
    end

    # Projection phase.
    project_conv = Conv(
        (1, 1), mid_channels=>out_channels, pad=SamePad(), bias=false,
    )
    bn2 = BatchNorm(out_channels; momentum, ϵ)
    projection = Chain(project_conv, bn2)

    MBConv(
        expansion, depthwise, excitation, projection, drop,
        do_expansion, do_excitation, do_skip, nothing,
    )
end

function Flux.testmode!(m::MBConv, mode = true)
    m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode
    m
end

function (m::MBConv)(x; drop_probability::Union{Float32, Nothing} = nothing)
    if m.do_expansion
        o = x |> m.expansion |> m.depthwise
    else
        o = x |> m.depthwise
    end

    if m.do_excitation
        o = σ.(o |> m.excitation) .* o
    end
    o = o |> m.projection |> copy # TODO remove copy when BatchNorm fixes Fill.Ones stuff with gradients

    if m.do_skip
        # The combination of skip connection and drop connect
        # brings about stochastic depth.
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
        "- active: ", m.active, "\n",
    )
end
