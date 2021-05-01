module EfficientNet
export EffNet, MBConv, get_model_params

using FileIO
using Images
using Pickle
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

function (m::EffNet)(x)
    o = x |> m.stem
    for (i, block) in enumerate(m.blocks)
        p = m.drop_connect
        p = isnothing(p) ? p : p * (i - 1) / length(m.blocks)
        o = block(o, drop_probability=p)
    end
    o = o |> m.head |> m.pooling
    m.top ≢ nothing && (o = o |> flatten |> m.top;)
    o
end

function _load_stem!(model::EffNet, params)
    rebuild_conv!(model.stem[1].weight, params["_conv_stem.weight"])
    copyto!(model.stem[2].γ, params["_bn0.weight"])
    copyto!(model.stem[2].β, params["_bn0.bias"])
    copyto!(model.stem[2].μ, params["_bn0.running_mean"])
    copyto!(model.stem[2].σ², params["_bn0.running_var"])
end

function _load_block!(block::MBConv, params, base)
    # expansion
    if block.expansion ≢ nothing
        rebuild_conv!(
            block.expansion[1].weight, params[base * "._expand_conv.weight"],
        )
        copyto!(block.expansion[2].γ, params[base * "._bn0.weight"])
        copyto!(block.expansion[2].β, params[base * "._bn0.bias"])
        copyto!(block.expansion[2].μ, params[base * "._bn0.running_mean"])
        copyto!(block.expansion[2].σ², params[base * "._bn0.running_var"])
    end

    # depthwise
    rebuild_conv!(
        block.depthwise[1].weight, params[base * "._depthwise_conv.weight"],
    )
    copyto!(block.depthwise[2].γ, params[base * "._bn1.weight"])
    copyto!(block.depthwise[2].β, params[base * "._bn1.bias"])
    copyto!(block.depthwise[2].μ, params[base * "._bn1.running_mean"])
    copyto!(block.depthwise[2].σ², params[base * "._bn1.running_var"])

    # excitation
    if block.excitation ≢ nothing
        rebuild_conv!(
            block.excitation[2].weight, params[base * "._se_reduce.weight"],
        )
        copyto!(block.excitation[2].bias, params[base * "._se_reduce.bias"])
        rebuild_conv!(
            block.excitation[4].weight, params[base * "._se_expand.weight"],
        )
        copyto!(block.excitation[4].bias, params[base * "._se_expand.bias"])
    end

    # projection
    rebuild_conv!(
        block.projection[1].weight, params[base * "._project_conv.weight"],
    )
    copyto!(block.projection[2].γ, params[base * "._bn2.weight"])
    copyto!(block.projection[2].β, params[base * "._bn2.bias"])
    copyto!(block.projection[2].μ, params[base * "._bn2.running_mean"])
    copyto!(block.projection[2].σ², params[base * "._bn2.running_var"])
end

function _load_blocks!(model::EffNet, params)
    for i in 1:length(model.blocks)
        _load_block!(model.blocks[i], params, "_blocks.$(i - 1)")
    end
end

function from_pretrained(model_name::String)
    block_params, global_params = get_model_params("efficientnet-b0")
    model = EffNet(block_params, global_params)

    w = Pickle.Torch.THload(raw"C:\Users\tonys\.cache\torch\hub\checkpoints\efficientnet-b0-355c32eb.pth")
    wkeys = w |> keys |> collect

    _load_stem!(model, w)
    _load_blocks!(model, w)

    rebuild_conv!(model.head[1].weight, w["_conv_head.weight"])
    copyto!(model.head[2].γ, w["_bn1.weight"])
    copyto!(model.head[2].β, w["_bn1.bias"])
    copyto!(model.head[2].μ, w["_bn1.running_mean"])
    copyto!(model.head[2].σ², w["_bn1.running_var"])

    if model.top ≢ nothing
        copyto!(model.top[2].weight, w["_fc.weight"])
        copyto!(model.top[2].bias, w["_fc.bias"])
    end
    model
end

function main()
    """
    TODO
    - auto download weights
    - check that training works
    """
    model = from_pretrained("efficientnet-b0")
    model = model |> testmode! |> gpu

    images = [
        raw"C:\Users\tonys\Downloads\elephant.png",
        raw"C:\Users\tonys\Downloads\bee.png",
        raw"C:\Users\tonys\Downloads\dog.png",
    ]
    for image in images
        x = Images.load(image) |> channelview .|> Float32
        x .-= reshape([0.485, 0.456, 0.406], (3, 1, 1))
        x ./= reshape([0.229, 0.224, 0.225], (3, 1, 1))
        x = Flux.unsqueeze(permutedims(x, (3, 2, 1)), 4)

        o = x |> gpu |> model |> softmax |> cpu
        o = sortperm(o[:, 1])
        @info o[end - 5:end]
    end
end
main()

end
