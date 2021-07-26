function round_filter(filters::Int, global_params::GlobalParams)::Int
    global_params.width_coef ≈ 1 && return filters

    depth_divisor = global_params.depth_divisor
    filters *= global_params.width_coef
    min_depth = global_params.min_depth
    min_depth = min_depth ≡ nothing ? depth_divisor : min_depth

    new_filters = max(
        min_depth,
        (floor(filters + depth_divisor / 2) ÷ depth_divisor) * depth_divisor
    )
    new_filters < 0.9 * filters && (new_filters += global_params.depth_divisor)
    new_filters
end

@inline function compute_output_image_size(
    image_size::Tuple{Int, Int}, stride::Int,
)
    ceil.(Int, image_size ./ stride)
end
