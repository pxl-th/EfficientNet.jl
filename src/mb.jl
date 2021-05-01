mutable struct MBConv{E, D, X, P}
    expansion::E
    depthwise::D
    excitation::X
    projection::P

    do_expansion::Bool
    do_excitation::Bool

    in_channels::Int64
    out_channels::Int64
    stride::Int64
    skip_connection::Bool

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
    mid_channels = ceil(Int, in_channels * expansion_ratio)
    do_expansion = expansion_ratio != 1
    do_excitation = 0 < se_ratio ≤ 1
    expansion, excitation = nothing, nothing

    activation = x -> x .|> swish

    # Expansion phase.
    if do_expansion
        expand_conv = Conv(
            (1, 1), in_channels=>mid_channels, bias=false, pad=SamePad(),
        )
        bn0 = BatchNorm(mid_channels, momentum=momentum, ϵ=ϵ)
        expansion = Chain(expand_conv, bn0, activation)
    end

    # Depthwise convolution phase.
    depthwise_conv = DepthwiseConv(
        kernel, mid_channels=>mid_channels,
        bias=false, stride=stride, pad=SamePad(),
    )
    bn1 = BatchNorm(mid_channels, momentum=momentum, ϵ=ϵ)
    depthwise = Chain(depthwise_conv, bn1, activation)

    # Squeeze and Excitation phase.
    if do_excitation
        n_squeezed_channels = max(1, ceil(Int, in_channels * se_ratio))
        squeeze_conv = Conv(
            (1, 1), mid_channels=>n_squeezed_channels, pad=SamePad(),
        )
        excite_conv = Conv(
            (1, 1), n_squeezed_channels=>mid_channels, pad=SamePad(),
        )
        excitation = Chain(
            AdaptiveMeanPool((1, 1)), squeeze_conv, activation, excite_conv,
        )
    end

    # Pointwise convolution phase.
    project_conv = Conv(
        (1, 1), mid_channels=>out_channels, pad=SamePad(), bias=false,
    )
    bn2 = BatchNorm(out_channels, momentum=momentum, ϵ=ϵ)
    projection = Chain(project_conv, bn2)

    MBConv(
        expansion, depthwise, excitation, projection,
        do_expansion, do_excitation,
        in_channels, out_channels, stride, skip_connection,
        nothing,
    )
end

function Flux.testmode!(m::MBConv, mode = true)
    m.active = (isnothing(mode) || mode == :auto) ? nothing : !mode
    m
end

function (m::MBConv)(x; drop_probability::Union{Float32, Nothing} = nothing)
    o = x
    m.do_expansion && (o = o |> m.expansion;)
    o = o |> m.depthwise
    m.do_excitation && (o = σ.(o |> m.excitation) .* o;)
    o = o |> m.projection

    if m.skip_connection && m.stride == 1 && m.in_channels == m.out_channels
        # The combination of skip connection and drop connect
        # brings about stochastic depth.
        if drop_probability ≢ nothing
            o = drop_connect(o, drop_probability, m |> Flux._isactive)
        end
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
        "- in channels: ", m.in_channels, "\n",
        "- out channels: ", m.out_channels, "\n",
        "- stride: ", m.stride, "\n",
        "- skip connections: ", m.skip_connection, "\n",
        "- active: ", m.active, "\n",
    )
end
