function round_filter(filters::Int64, global_params::GlobalParams)::Int64
    global_params.width_coef ≈ 1 && return filters

    filters *= global_params.width_coef
    min_depth = (
        global_params.min_depth ≡ nothing
        ? global_params.depth_divisor
        : global_params.min_depth
    )
    new_filters = max(
        min_depth,
        (filters + global_params.depth_divisor ÷ 2)
        ÷ global_params.depth_divisor
        * global_params.depth_divisor
    )
    new_filters > 0.9 * filters && (new_filters += global_params.depth_divisor)
    new_filters
end

@inline function compute_output_image_size(
    image_size::Tuple{Int64, Int64}, stride::Int64,
)
    ceil.(Int64, image_size ./ stride)
end
