struct EffNet{S, B, H, P, F}
    stem::S
    blocks::B
    head::H
    pooling::P
    top::F

    drop_connect::Union{Float32, Nothing}
end

Flux.@functor EffNet

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

    EffNet(stem, blocks, head, avg_pool, top, global_params.drop_connect_rate)
end

EffNet(model_name::String; kwargs...) = get_model_params(model_name; kwargs...)

function (m::EffNet)(x)
    o = x |> m.stem
    for (i, block) in enumerate(m.blocks)
        p = m.drop_connect
        p = isnothing(p) ? p : p * (i - 1) / length(m.blocks)
        o = block(o; drop_probability=p)
    end
    o = o |> m.head |> m.pooling
    m.top ≢ nothing && (o = o |> flatten |> m.top;)
    o
end

"""
Use convolution layers to extract features from reduction levels.
"""
function extract(m::EffNet, x)
    endpoints = []

    o = x |> m.stem
    prev_o = o
    for (i, block) in enumerate(m.blocks)
        p = m.drop_connect
        p = isnothing(p) ? p : p * (i - 1) / length(m.blocks)
        o = block(o; drop_probability=p)

        # Add endpoint if decreased resolution or it is the last block.
        if size(prev_o, 1) > size(o, 1)
            push!(endpoints, prev_o)
        elseif i == length(m.blocks)
            push!(endpoints, o)
        end
        prev_o = o
    end

    push!(endpoints, o |> m.head)
    endpoints
end
