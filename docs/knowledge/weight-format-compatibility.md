# Weight Format Compatibility

This document describes weight compatibility between different device configurations in the MSI-Net model.

## Data Formats

The model uses two data formats depending on the device:

| Device | Data Format | Tensor Shape |
|--------|-------------|--------------|
| `gpu` (CUDA) | `channels_first` (NCHW) | `[batch, channels, height, width]` |
| `cpu` | `channels_last` (NHWC) | `[batch, height, width, channels]` |
| `tpu` | `channels_last` (NHWC) | `[batch, height, width, channels]` |
| `metal` | `channels_last` (NHWC) | `[batch, height, width, channels]` |

## Compatibility Matrix

| Trained With | Can Inference On | Notes |
|--------------|------------------|-------|
| `gpu` | `gpu` only | CUDA weights use `channels_first` |
| `cpu` | `cpu`, `tpu`, `metal` | All use `channels_last` |
| `tpu` | `cpu`, `tpu`, `metal` | All use `channels_last` |
| `metal` | `cpu`, `tpu`, `metal` | All use `channels_last` |

## Why Weights Are Not Cross-Compatible

The model's convolutional layers store weights in a format tied to the data layout:

- `channels_first` (NCHW): Optimized for NVIDIA CUDA
- `channels_last` (NHWC): Required by TPU and Metal, also the TensorFlow default

Weight conversion between formats is not straightforward because the weight tensors themselves have different shapes and interpretations depending on the data format used during training.

## Recommendations

1. **For Linux deployment**: If training on Mac with Metal and deploying on Linux:
   - Train with `device: "metal"`
   - On Linux, use `device: "cpu"` for inference
   - This works because both use `channels_last`

2. **For maximum compatibility**: Train with `device: "cpu"` or any `channels_last` device (tpu, metal)

3. **For NVIDIA GPU performance**: If you need CUDA GPU performance on Linux, train with `device: "gpu"` and keep using `device: "gpu"` for inference

## Weight File Naming

Weights are saved with the device suffix to identify format:
- `model_salicon_gpu.weights.h5` - CUDA/channels_first
- `model_salicon_cpu.weights.h5` - CPU/channels_last
- `model_salicon_tpu.weights.h5` - TPU/channels_last
- `model_salicon_metal.weights.h5` - Metal/channels_last
