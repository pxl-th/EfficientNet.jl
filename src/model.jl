struct EffNet{S, B, H, P, F}
    stem::S
    blocks::B
    head::H
    pooling::P
    top::F

    drop_connect::Union{Real, Nothing}
    model_name::String
end

Flux.@functor EffNet

function EffNet(
    model_name::String,
    block_params::Vector{BlockParams}, global_params::GlobalParams,
    include_head::Bool = true,
)
    # Stem.
    out_channels = round_filter(32, global_params)
    stem_conv = Conv(
        (3, 3), 3=>out_channels, bias=false, stride=2, pad=SamePad(),
    )
    stem_bn = BatchNorm(
        out_channels, swish,
        # momentum=global_params.bn_momentum, ϵ=global_params.bn_ϵ,
    )
    stem = Chain(stem_conv, stem_bn)
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

    include_head || return EffNet(
        stem, blocks, nothing, nothing, nothing,
        global_params.drop_connect_rate, model_name,
    )

    # Head.
    head_out_channels = round_filter(1280, global_params)
    head_conv = Conv(
        (1, 1), out_channels=>head_out_channels, bias=false, pad=SamePad(),
    )
    head_bn = BatchNorm(
        head_out_channels, swish,
        # momentum=global_params.bn_momentum, ϵ=global_params.bn_ϵ,
    )
    head = Chain(head_conv, head_bn)
    # Final linear.
    avg_pool = AdaptiveMeanPool((1, 1))
    top = nothing
    if global_params.include_top
        dropout = Dropout(global_params.dropout_rate)
        fc = Dense(head_out_channels, global_params.n_classes)
        top = Chain(dropout, fc)
    end

    EffNet(
        stem, blocks, head, avg_pool, top,
        global_params.drop_connect_rate, model_name,
    )
end

EffNet(model_name::String; kwargs...) =
    EffNet(model_name, get_model_params(model_name; kwargs...)...)

function (m::EffNet)(x)
    o = x |> m.stem
    for (i, block) in enumerate(m.blocks)
        p = m.drop_connect
        if !isnothing(p)
            p = p * (i - 1) / length(m.blocks)
        end
        o = block(o; drop_probability=p)
    end
    if m.head ≡ nothing
        return o
    end
    o = o |> m.head |> m.pooling
    if m.top ≡ nothing
        return o
    end
    o |> flatten |> m.top
end

"""
Use convolution layers to extract features from reduction levels.
"""
function (m::EffNet)(x, ::Val{:stages})
    sid = 1
    stages_ids = m |> get_stages
    stages = [x]

    o = x |> m.stem
    push!(stages, o)

    for (i, block) in enumerate(m.blocks)
        p = m.drop_connect
        if !isnothing(p)
            p = p * (i - 1) / length(m.blocks)
        end
        o = block(o; drop_probability=p)

        if i == stages_ids[sid]
            sid += 1
            push!(stages, o)
            if sid > length(stages_ids)
                break
            end
        end
    end

    stages
end

function (m::EffNet)(x::V, ::Val{:stages_map}) where V <: AbstractArray
    # Create stages to pass to `map`.
    sids = m |> get_stages
    stages = (
        m.stem, m.blocks[1:sids[1]],
        m.blocks[sids[1] + 1:sids[2]],
        m.blocks[sids[2] + 1:sids[3]],
        m.blocks[sids[3] + 1:sids[4]],
    )

    block_id = 0
    inv_length = 1.0 / length(m.blocks)
    # Define `runner` function, to map over `stages`.
    runner(block::Chain)::V = (x = block(x); x)
    function runner(blocks::T)::V where T <: AbstractVector
        for block in blocks
            p = m.drop_connect
            if p ≢ nothing
                p = p * block_id * inv_length
            end
            x = block(x; drop_probability=p)
            block_id += 1
        end
        x
    end

    map(runner, stages)
end

function get_stages(e::EffNet)
    d = Dict(
        "efficientnet-b0" => [3, 5, 9, 16],
        "efficientnet-b1" => [5, 8, 16, 23],
        "efficientnet-b2" => [5, 8, 16, 23],
        "efficientnet-b3" => [5, 8, 18, 26],
        "efficientnet-b4" => [6, 10, 22, 32],
        "efficientnet-b5" => [8, 13, 27, 39],
        "efficientnet-b6" => [9, 15, 31, 45],
        "efficientnet-b7" => [11, 18, 38, 55],
    )
    if !(e.model_name in keys(d))
        throw("Only `efficientnet-[b0-b7]` are supported.")
    end
    d[e.model_name]
end

function stages_channels(e::EffNet)
    d = Dict(
        "efficientnet-b0" => (32, 24, 40, 112, 320),
        "efficientnet-b1" => (32, 24, 40, 112, 320),
        "efficientnet-b2" => (32, 24, 48, 120, 352),
        "efficientnet-b3" => (40, 32, 48, 136, 384),
        "efficientnet-b4" => (48, 32, 56, 160, 448),
        "efficientnet-b5" => (48, 40, 64, 176, 512),
        "efficientnet-b6" => (56, 40, 72, 200, 576),
        "efficientnet-b7" => (64, 48, 80, 224, 640),
    )
    if !(e.model_name in keys(d))
        throw("Only `efficientnet-[b0-b7]` are supported.")
    end
    d[e.model_name]
end
