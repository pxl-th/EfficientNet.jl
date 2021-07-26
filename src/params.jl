struct BlockParams
    repeat::Int
    kernel::Tuple{Int, Int}
    stride::Int
    expand_ratio::Int
    in_channels::Int
    out_channels::Int
    se_ratio::Real
    skip_connection::Bool
end

struct GlobalParams
    width_coef::Real
    depth_coef::Real
    image_size::Tuple{Int, Int}
    dropout_rate::Real

    n_classes::Int
    bn_momentum::Real
    bn_ϵ::Real

    drop_connect_rate::Real
    depth_divisor::Int
    min_depth::Union{Nothing, Int}
    include_top::Bool
end

function get_efficientnet_params(model_name::String)
    # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    Dict(
      "efficientnet-b0" => (1.0, 1.0, 224, 0.2),
      "efficientnet-b1" => (1.0, 1.1, 240, 0.2),
      "efficientnet-b2" => (1.1, 1.2, 260, 0.3),
      "efficientnet-b3" => (1.2, 1.4, 300, 0.3),
      "efficientnet-b4" => (1.4, 1.8, 380, 0.4),
      "efficientnet-b5" => (1.6, 2.2, 456, 0.4),
      "efficientnet-b6" => (1.8, 2.6, 528, 0.5),
      "efficientnet-b7" => (2.0, 3.1, 600, 0.5),
      "efficientnet-b8" => (2.2, 3.6, 672, 0.5),
      "efficientnet-l2" => (4.3, 5.3, 800, 0.5),
    )[model_name]
end

function get_model_params(
    model_name::String;
    n_classes::Int = 1000,
    drop_connect_rate::Real = 0.2,
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
