struct BlockParams
    repeat::Int64
    kernel::Tuple{Int64, Int64}
    stride::Int64
    expand_ratio::Int64
    in_channels::Int64
    out_channels::Int64
    se_ratio::Float64
    skip_connection::Bool
end

struct GlobalParams
    width_coef::Float64
    depth_coef::Float64
    image_size::Tuple{Int64, Int64}
    dropout_rate::Float32

    n_classes::Int64
    bn_momentum::Float32
    bn_ϵ::Float32

    drop_connect_rate::Float32
    depth_divisor::Int64
    min_depth::Union{Nothing, Int64}
    include_top::Bool
end

function get_efficientnet_params(model_name::String)
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    Dict(
      "efficientnet-b0" => (1.0, 1.0, 224, 0.2f0),
      "efficientnet-b1" => (1.0, 1.1, 240, 0.2f0),
      "efficientnet-b2" => (1.1, 1.2, 260, 0.3f0),
      "efficientnet-b3" => (1.2, 1.4, 300, 0.3f0),
      "efficientnet-b4" => (1.4, 1.8, 380, 0.4f0),
      "efficientnet-b5" => (1.6, 2.2, 456, 0.4f0),
      "efficientnet-b6" => (1.8, 2.6, 528, 0.5f0),
      "efficientnet-b7" => (2.0, 3.1, 600, 0.5f0),
      "efficientnet-b8" => (2.2, 3.6, 672, 0.5f0),
      "efficientnet-l2" => (4.3, 5.3, 800, 0.5f0),
    )[model_name]
end

function get_model_params(
    model_name::String;
    n_classes::Int64 = 1000,
    drop_connect_rate::Float32 = 0.2f0,
    include_top::Bool = true
)
    block_params = [
        BlockParams(1, (3, 3), 1, 1,  32,  16, 0.25, true),
        BlockParams(2, (3, 3), 2, 6,  16,  24, 0.25, true),
        BlockParams(2, (5, 5), 2, 6,  24,  40, 0.25, true),
        BlockParams(3, (3, 3), 2, 6,  40,  80, 0.25, true),
        BlockParams(3, (5, 5), 1, 6,  80, 112, 0.25, true),
        BlockParams(4, (5, 5), 2, 6, 112, 192, 0.25, true),
        BlockParams(1, (3, 3), 1, 6, 192, 320, 0.25, true),
    ]

    wc, dc, res, drop = get_efficientnet_params(model_name)
    global_params = GlobalParams(
        wc, dc, (res, res), drop,
        n_classes, 0.99f0, 1f-3,
        drop_connect_rate, 8, nothing, include_top,
    )
    block_params, global_params
end
