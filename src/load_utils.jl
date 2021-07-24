# Pytorch loading utils.

function rebuild_conv!(dst, src)
    shape = dst |> size
    filter_x, filter_y = shape[1:2] .+ 1
    for (i, j, k, m) in Iterators.product([1:s for s in shape]...)
        dst[filter_x - i, filter_y - j, k, m] = src[m, k, j, i]
    end
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
            block.excitation[3].weight, params[base * "._se_expand.weight"],
        )
        copyto!(block.excitation[3].bias, params[base * "._se_expand.bias"])
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

function _load_head!(model::EffNet, params)
    rebuild_conv!(model.head[1].weight, params["_conv_head.weight"])
    copyto!(model.head[2].γ, params["_bn1.weight"])
    copyto!(model.head[2].β, params["_bn1.bias"])
    copyto!(model.head[2].μ, params["_bn1.running_mean"])
    copyto!(model.head[2].σ², params["_bn1.running_var"])

    if model.top ≢ nothing
        copyto!(model.top[2].weight, params["_fc.weight"])
        copyto!(model.top[2].bias, params["_fc.bias"])
    end
end

@inline function _load_pth!(model::EffNet, params, include_head::Bool)
    _load_stem!(model, params)
    _load_blocks!(model, params)
    include_head && _load_head!(model, params)
end

function from_pretrained(
    model_name::String; cache_dir::Union{String, Nothing} = nothing,
    include_head::Bool = true,
)
    url_base = (
        "https://github.com/lukemelas/EfficientNet-PyTorch" *
        "/releases/download/1.0/"
    )
    url_map = Dict(
        "efficientnet-b0" => "efficientnet-b0-355c32eb.pth",
        "efficientnet-b1" => "efficientnet-b1-f1951068.pth",
        "efficientnet-b2" => "efficientnet-b2-8bb594d6.pth",
        "efficientnet-b3" => "efficientnet-b3-5fb5a3c3.pth",
        "efficientnet-b4" => "efficientnet-b4-6ed6700e.pth",
        "efficientnet-b5" => "efficientnet-b5-b6417697.pth",
        "efficientnet-b6" => "efficientnet-b6-c76e70fd.pth",
        "efficientnet-b7" => "efficientnet-b7-dcc49843.pth",
    )
    if !(model_name in keys(url_map))
        error(
            "Invalid model name: $model_name. " *
            "Supported pretrained models: $(keys(url_map)) "
        )
    end

    if cache_dir ≡ nothing
        cache_dir = joinpath(homedir(), ".cache", "EfficientNet.jl")
        !isdir(cache_dir) && mkdir(cache_dir)
        @info "Using default cache dir $cache_dir"
    end

    params_file = url_map[model_name]
    params_path = joinpath(cache_dir, params_file)
    if !isfile(params_path)
        download_url = url_base * params_file
        @info(
            "Downloading $model_name params:\n" *
            "\t- from URL: $download_url \n" *
            "\t- to directory: $params_path"
        )
        download(download_url, params_path)
        @info "Finished downloading params."
    end

    block_params, global_params = get_model_params(model_name)
    model = EffNet(model_name, block_params, global_params, include_head)

    params = Pickle.Torch.THload(params_path)
    _load_pth!(model, params, include_head)

    model
end
