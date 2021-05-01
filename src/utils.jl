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

function drop_connect(x::AbstractArray{T}, p, active::Bool) where T <: Number
    !active && return x
    keep_p = T(1 - p)
    ϵ = floor.(keep_p .+ _randlike(x))
    x ./ keep_p .* ϵ
end

_randlike(x::AbstractArray{T}) where T = randn(T, (1, 1, 1, size(x, 4)))
_randlike(x::CuArray{T}) where T = CUDA.randn(T, (1, 1, 1, size(x, 4)))

function rebuild_conv!(dst, src)
    shape = dst |> size
    filter_x, filter_y = shape[1:2] .+ 1
    for (i, j, k, m) in Iterators.product([1:s for s in shape]...)
        @inbounds dst[filter_x - i, filter_y - j, k, m] = src[m, k, j, i]
    end
end
