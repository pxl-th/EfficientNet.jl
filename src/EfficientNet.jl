module EfficientNet

using CUDA
using Flux

include("params.jl")
include("mb.jl")
include("utils.jl")

struct EffNet{S, B, H, P, F}
    stem::S
    blocks::B
    head::H
    pooling::P
    final::F
end

function EffNet(
    block_params::Vector{BlockParams}, global_params::GlobalParams,
)
    activation = x -> x .|> swish

    # Stem.
    out_channels = round_filter(32, global_params)
    stem_conv = Conv(
        (3, 3), 3=>out_channels, bias=false, stride=2, pad=SamePad(),
    )
    stem_bn = BatchNorm(
        out_channels, momentum=global_params.bn_momentum, ϵ=global_params.bn_ϵ,
    )
    stem = Chain(stem_conv, stem_bn, activation)

    # Build blocks.
    blocks = MBConv[]
    for bp in block_params
        in_channels = round_filter(bp.in_channels, global_params)
        out_channels = round_filter(bp.out_channels, global_params)
        repeat = (
            global_params.depth_coef ≈ 1
            ? bp.repeat
            : ceil(Int64, global_params.depth_coef * bp.repeat)
        )

        push!(blocks, MBConv(
            in_channels, out_channels, bp.kernel, bp.stride,
            expansion_ratio=bp.expand_ratio, se_ratio=bp.se_ratio,
            skip_connection=bp.skip_connection,
            momentum=global_params.bn_momentum, ϵ=global_params.bn_ϵ,
        ))
        for _ in 1:(repeat - 1)
            push!(blocks, MBConv(
                out_channels, out_channels, bp.kernel, 1,
                expansion_ratio=bp.expand_ratio, se_ratio=bp.se_ratio,
                skip_connection=bp.skip_connection,
                momentum=global_params.bn_momentum, ϵ=global_params.bn_ϵ,
            ))
        end
    end

    # Head.
    head_out_channels = round_filter(1280, global_params)
    head_conv = Conv(
        (1, 1), out_channels=>head_out_channels, bias=false, pad=SamePad(),
    )
    head_bn = BatchNorm(
        head_out_channels, momentum=global_params.bn_momentum,
        ϵ=global_params.bn_ϵ,
    )
    head = Chain(head_conv, head_bn, activation)

    # Final linear.
    avg_pool = AdaptiveMeanPool((1, 1))
    top = nothing
    if global_params.include_top
        dropout = Dropout(global_params.dropout_rate)
        fc = Dense(head_out_channels, global_params.n_classes)
        top = Chain(dropout, fc)
    end

    EffNet(stem, blocks, head, avg_pool, top)
end

function (m::EffNet)(x)
    o = x |> m.stem
    for block in m.blocks
        # TODO add drop connect rate
        o = block(o)
    end
    o = o |> m.head |> m.pooling
    @info typeof(o)
    m.final ≢ nothing && (o = o |> flatten |> m.final;)
    @info typeof(o)
    o
end

block_params, global_params = get_model_params("efficientnet-b0")
model = EffNet(block_params, global_params)

# m = MBConv(
#     1, 1, (3, 3), 1,
#     expansion_ratio=2f0, se_ratio=0.5f0, skip_connection=false,
#     momentum=0.99f0, ϵ=1f-6,
# )
# @show m

x = randn(Float32, (224, 224, 3, 1))
o = x |> model
println(typeof(x), typeof(o))
println(size(x), size(o))

end
