# EfficientNet.jl

EfficientNet implementation in Julia.

## Install

```
]add https://github.com/pxl-th/EfficientNet.jl.git
```

## Construct model

```julia
model = EffNet("efficientnet-b0"; n_classes=10, in_channels=3)
```

## ImageNet pretrained model

```julia
model = EfficientNet.from_pretrained("efficientnet-b3")
```

By default, weights are stored in `~/.cache/EfficientNet.jl/` directory.
If there are no weights, it will attempt to download them.
Additionally, you can specify cache directory as a second parameter to `from_pretrained` function.

Available pretrained models are B0-B7.
They are loaded from [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
and converted to Julia's format.

## Example inference on image

```julia
using Images
using EfficientNet

device = gpu
model = EfficientNet.from_pretrained("efficientnet-b3")
model = model |> testmode! |> device

image = "./spaceshuttle.png"
x = Images.load(image) |> channelview .|> Float32
x = Flux.unsqueeze(permutedims(x, (3, 2, 1)), 4)

o = x |> device |> model |> softmax |> cpu
o = sortperm(o[:, 1])
@info "Top 5 classes: $(o[end:-1:end - 5] .- 1)"
```

## Feature extraction

To extract list of features, pass `Val(:stages)` as the second parameter:

```julia
features = model(x, Val(:stages))
```

It will contain features, extracted from different resolution levels.
This can be used in something like UNet architecture as an encoder.
To get those resolution levels, use `model.stages` and `model.stages_channels`
to get ids of specific layers and output channels.
