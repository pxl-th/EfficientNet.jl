struct EffNet{S, B, H, P, F}
    stem::S
    blocks::B

    head::H
    pooling::P
    top::F

    drop_connect::Union{Real, Nothing}
    stages::NTuple{4, Int}
    stages_channels::NTuple{5, Int}
end
Flux.@functor EffNet

function EffNet(model_name, block_params, global_params; include_head = true, in_channels = 3)
    pad, bias = SamePad(), false
    out_channels = round_filter(32, global_params)
    stem = Chain(Conv((3, 3), in_channels=>out_channels; bias, stride=2, pad), BatchNorm(out_channels, swish))

    blocks = MBConv[]
    for bp in block_params
        in_channels = round_filter(bp.in_channels, global_params)
        out_channels = round_filter(bp.out_channels, global_params)
        repeat = global_params.depth_coef ≈ 1 ? bp.repeat : ceil(Int64, global_params.depth_coef * bp.repeat)

        expansion_ratio, se_ratio, skip_connection = bp.expand_ratio, bp.se_ratio, bp.skip_connection
        momentum, ϵ = global_params.bn_momentum, global_params.bn_ϵ
        push!(blocks, MBConv(in_channels, out_channels, bp.kernel, bp.stride; expansion_ratio, se_ratio, skip_connection, momentum, ϵ))
        for _ in 1:(repeat - 1)
            push!(blocks, MBConv(out_channels, out_channels, bp.kernel, 1; expansion_ratio, se_ratio, skip_connection, momentum, ϵ))
        end
    end

    stages = get_stages(model_name)
    channels = stages_channels(model_name)
    include_head || return EffNet(stem, blocks, nothing, nothing, nothing, global_params.drop_connect_rate, stages, channels)

    head_out_channels = round_filter(1280, global_params)
    head = Chain(Conv((1, 1), out_channels=>head_out_channels; bias, pad), BatchNorm(head_out_channels, swish))
    top = global_params.include_top ? Chain(Dropout(global_params.dropout_rate), Dense(head_out_channels, global_params.n_classes)) : nothing
    EffNet(stem, blocks, head, AdaptiveMeanPool((1, 1)), top, global_params.drop_connect_rate, stages, channels)
end

EffNet(model_name::String; include_head = true, in_channels = 3, kwargs...) =
    EffNet(model_name, get_model_params(model_name; kwargs...)...; include_head, in_channels)

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

function (m::EffNet)(x::V, ::Val{:stages}) where V <: AbstractArray
    block_id, inv_length = 0, 1.0 / length(m.blocks)
    stages = (
        m.stem, m.blocks[1:m.stages[1]], m.blocks[(m.stages[1] + 1):m.stages[2]],
        m.blocks[(m.stages[2] + 1):m.stages[3]], m.blocks[(m.stages[3] + 1):m.stages[4]])

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

get_stages(model_name) =
    Dict(
        "efficientnet-b0" => (3, 5, 9, 16),
        "efficientnet-b1" => (5, 8, 16, 23),
        "efficientnet-b2" => (5, 8, 16, 23),
        "efficientnet-b3" => (5, 8, 18, 26),
        "efficientnet-b4" => (6, 10, 22, 32),
        "efficientnet-b5" => (8, 13, 27, 39),
        "efficientnet-b6" => (9, 15, 31, 45),
        "efficientnet-b7" => (11, 18, 38, 55))[model_name]
stages_channels(model_name) =
    Dict(
        "efficientnet-b0" => (32, 24, 40, 112, 320),
        "efficientnet-b1" => (32, 24, 40, 112, 320),
        "efficientnet-b2" => (32, 24, 48, 120, 352),
        "efficientnet-b3" => (40, 32, 48, 136, 384),
        "efficientnet-b4" => (48, 32, 56, 160, 448),
        "efficientnet-b5" => (48, 40, 64, 176, 512),
        "efficientnet-b6" => (56, 40, 72, 200, 576),
        "efficientnet-b7" => (64, 48, 80, 224, 640))[model_name]
