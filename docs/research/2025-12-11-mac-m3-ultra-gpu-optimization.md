---
date: 2025-12-11T00:00:00+00:00
researcher: Claude
topic: "Running MSI-Net on Mac Studio M3 Ultra GPU"
tags: [research, apple-silicon, m3-ultra, tensorflow-metal, gpu-optimization, mlx]
last_updated: 2025-12-11
last_updated_by: Claude
last_updated_note: "Added MLX framework investigation"
---

# Research: Running MSI-Net on Mac Studio M3 Ultra GPU

## Research Question

I want to run this code on a mac studio with an m3 ultra, what changes need to be made to optimise this for running on its GPU

## Summary

The MSI-Net saliency model codebase currently uses `channels_first` (NCHW) data format for GPU mode, which is optimized for NVIDIA CUDA but is **not compatible** with Apple's tensorflow-metal plugin. Running on Apple Silicon requires using `channels_last` (NHWC) format, which the codebase already supports through its CPU/TPU mode. The key finding is that the existing TPU code path (`device: "tpu"` or `device: "cpu"`) uses the correct data format for Apple Metal GPU acceleration.

---

## Detailed Findings

### 1. Current Codebase GPU Configuration

#### 1.1 Device Selection Mechanism

**Location:** `config.py:13-18`

```python
PARAMS = {
    "n_epochs": 10,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "device": "gpu"
}
```

The codebase uses a single `device` parameter that controls the data format and model optimization throughout the entire training pipeline.

#### 1.2 Data Format Selection

**Location:** `model.py:47-55`

```python
if config.PARAMS["device"] == "gpu":
    self._data_format = "channels_first"
    self._channel_axis = 1
    self._dims_axis = (2, 3)
elif config.PARAMS["device"] in ("cpu", "tpu"):
    # TPU requires channels_last format (same as CPU)
    self._data_format = "channels_last"
    self._channel_axis = 3
    self._dims_axis = (1, 2)
```

The model architecture dynamically configures all Conv2D and MaxPooling2D layers based on this setting:
- **GPU mode:** Uses `channels_first` (NCHW) - optimized for NVIDIA CUDA
- **CPU/TPU mode:** Uses `channels_last` (NHWC) - required for Apple Metal and Google TPU

#### 1.3 Layer Configuration

All convolutional and pooling layers in the MSINET class accept the `data_format` parameter:

**Location:** `model.py:58-144`

```python
self.conv1_1 = tf.keras.layers.Conv2D(
    64, 3, padding="same", activation="relu",
    data_format=self._data_format, name="conv1_conv1_1")
# ... (all 26 Conv2D layers follow this pattern)

self.pool1 = tf.keras.layers.MaxPooling2D(
    2, 2, data_format=self._data_format)
# ... (all 5 pooling layers follow this pattern)
```

#### 1.4 Tensor Format Conversion

The model includes explicit transpose operations for handling format differences during upsampling:

**Location:** `model.py:160-170`

```python
def _upsample(self, stack, target_shape, factor):
    if self._data_format == "channels_first":
        stack = tf.transpose(stack, (0, 2, 3, 1))
    # ... resize operation (always uses NHWC internally)
    if self._data_format == "channels_first":
        stack = tf.transpose(stack, (0, 3, 1, 2))
    return stack
```

---

### 2. TensorFlow-Metal Requirements and Limitations

#### 2.1 System Requirements

Based on Apple's official documentation:

| Requirement | Specification |
|-------------|---------------|
| Mac Hardware | Apple Silicon (M1, M2, M3, M4 series) or AMD GPU |
| macOS Version | 12.0 (Monterey) or later |
| Python Version | 3.9 - 3.11 (3.12 has limited support) |
| TensorFlow Version | 2.13+ (use standard `tensorflow`, not `tensorflow-macos` for 2.13+) |

#### 2.2 Installation

```bash
python -m pip install tensorflow
python -m pip install tensorflow-metal
```

#### 2.3 Known Limitations

1. **Data Format Limitation:** `channels_first` (NCHW) operations are not fully supported on Metal GPU. Operations fall back to CPU or fail with errors like "The Conv2D op currently only supports the NHWC tensor format on the CPU."

2. **Complex Data Types:** DT_COMPLEX64 and other complex types are not supported

3. **Multi-GPU:** Not supported on tensorflow-metal

4. **V1 Networks:** Legacy TensorFlow 1.x style networks are not supported

5. **Small Batch Performance:** CPU may outperform GPU for small networks with small batch sizes due to dispatch overhead

#### 2.4 M3 Specific Considerations

- TensorFlow 2.15-2.17 with tensorflow-metal 1.1.0-1.2.0 has been reported working on M3 chips
- Python 3.11 is the most stable version for tensorflow-metal
- Some M3 users have migrated to Apple MLX due to better unified memory support

---

### 3. Current Framework Dependencies

**Location:** `pyproject.toml:24-33`

```toml
dependencies = [
    "gdown>=5.0.0",
    "h5py>=3.10.0",
    "imageio>=2.33.0",
    "matplotlib>=3.8.0",
    "numpy>=1.26.0",
    "requests>=2.31.0",
    "scipy>=1.11.0",
    "tensorflow>=2.15.0",
]
```

The project currently specifies `tensorflow>=2.15.0` which is compatible with tensorflow-metal.

#### GPU Optional Dependency

```toml
[project.optional-dependencies]
gpu = [
    "tensorflow[and-cuda]>=2.15.0",
]
```

This GPU optional dependency targets NVIDIA CUDA, not Apple Metal.

---

### 4. Model Weight Compatibility

#### 4.1 Weight File Naming Convention

**Location:** `model.py:283-294`

```python
def save_weights(self, dataset, path, device):
    weights_path = path + "model_%s_%s.weights.h5" % (dataset, device)
    super().save_weights(weights_path)
```

Weights are saved with the device name embedded (e.g., `model_salicon_gpu.weights.h5`). This means:
- Weights trained with `device: "gpu"` (`channels_first`) cannot be directly loaded for `channels_last` inference
- The data format affects weight shape interpretation for convolutional layers

---

### 5. Existing Platform Support

#### 5.1 Current Device Options

The codebase currently supports three device modes:

| Device | Data Format | Target Platform |
|--------|-------------|-----------------|
| `gpu` | `channels_first` (NCHW) | NVIDIA CUDA GPUs |
| `cpu` | `channels_last` (NHWC) | Any CPU |
| `tpu` | `channels_last` (NHWC) | Google Cloud TPU |

#### 5.2 Vertex AI Integration

**Location:** `vertex-ai-config.yaml`

The codebase has existing cloud GPU training support for:
- NVIDIA Tesla T4
- NVIDIA Tesla A100
- Google TPU v5e

---

### 6. M3 Ultra Hardware Considerations

#### 6.1 M3 Ultra Specifications

| Specification | Value |
|---------------|-------|
| GPU Cores | Up to 80 |
| Unified Memory | Up to 512GB |
| Memory Bandwidth | 819.2 GB/s |
| Neural Engine | 32-core |

#### 6.2 Unified Memory Advantage

Apple's unified memory architecture eliminates the "Out of VRAM" issues common with discrete GPUs, though tensorflow-metal may not fully leverage this benefit compared to native frameworks like MLX.

---

### 7. Alternative Framework: Apple MLX

#### 7.1 MLX Overview

Apple MLX is a machine learning framework specifically designed for Apple Silicon, released by Apple Machine Learning Research in December 2023. It is designed to leverage Apple's hardware to its fullest potential, particularly the M-series chips.

**Key Characteristics:**

| Feature | MLX | TensorFlow (Metal) |
|---------|-----|-------------------|
| **Target Platform** | Apple Silicon only | Cross-platform with Metal backend |
| **Memory Model** | Unified memory (CPU/GPU shared) | Separate CPU/GPU memory |
| **Data Format** | NHWC (channels_last) only | Supports both NCHW and NHWC |
| **Computation** | Lazy evaluation | Eager or graph mode |
| **API Style** | NumPy-like core, PyTorch-like nn | Keras high-level API |
| **Ecosystem Maturity** | Young (2023+) | Mature (10+ years) |

#### 7.2 Unified Memory Advantage

A notable difference from MLX and other frameworks is the unified memory model. Arrays in MLX live in shared memory. Operations on MLX arrays can be performed on any of the supported device types without transferring data. This eliminates the notorious latency from CPU-GPU data transfers that affects TensorFlow.

#### 7.3 MLX Neural Network API for CNNs

Based on the MLX 0.30.0 documentation, the following layers are available:

**Conv2d:**
```python
class Conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple,
    stride: int | tuple = 1,
    padding: int | tuple = 0,
    dilation: int | tuple = 1,  # SUPPORTS DILATED CONVOLUTIONS
    groups: int = 1,
    bias: bool = True
)
```

**Critical Note:** MLX Conv2d expects **NHWC format** (channels last). This aligns with the current MSI-Net model's `channels_last` mode used for CPU/TPU.

**Dilation Support:** MLX Conv2d supports the `dilation` parameter, which is essential for the ASPP (Atrous Spatial Pyramid Pooling) module in MSI-Net that uses dilation rates of 2, 4, 8, and 12.

**Other Required Layers:**
- `MaxPool2d(kernel_size, stride=None, padding=0)` - 2D max pooling
- `Upsample(scale_factor, mode='nearest'|'linear'|'cubic')` - Bilinear upsampling supported via `mode='linear'`
- `mlx.optimizers.Adam(learning_rate)` - Adam optimizer

#### 7.4 MSI-Net Architecture Compatibility with MLX

| Component | TensorFlow | MLX Equivalent | Compatibility |
|-----------|------------|----------------|---------------|
| Conv2D with dilation | `tf.keras.layers.Conv2D(dilation_rate=...)` | `mlx.nn.Conv2d(dilation=...)` | FULL |
| MaxPooling2D | `tf.keras.layers.MaxPooling2D` | `mlx.nn.MaxPool2d` | FULL |
| Bilinear upsample | `tf.image.resize(..., method="bilinear")` | `mlx.nn.Upsample(mode='linear')` | FULL |
| Concatenation | `tf.concat()` | `mx.concatenate()` | FULL |
| Global avg pooling | `tf.reduce_mean()` | `mx.mean()` | FULL |
| Data format | channels_first/channels_last | channels_last only | COMPATIBLE* |

*The existing MSI-Net already supports `channels_last` mode for CPU/TPU, which aligns with MLX's requirements.

#### 7.5 Custom Loss Function in MLX

The KLD loss function can be implemented in MLX:

```python
# TensorFlow version (current)
def kld_loss(y_true, y_pred, eps=1e-7):
    sum_per_image = tf.reduce_sum(y_true, axis=(1, 2, 3), keepdims=True)
    y_true = y_true / (eps + sum_per_image)
    sum_per_image = tf.reduce_sum(y_pred, axis=(1, 2, 3), keepdims=True)
    y_pred = y_pred / (eps + sum_per_image)
    loss = y_true * tf.math.log(eps + y_true / (eps + y_pred))
    return tf.reduce_mean(tf.reduce_sum(loss, axis=(1, 2, 3)))

# MLX equivalent
def kld_loss(y_true, y_pred, eps=1e-7):
    sum_per_image = mx.sum(y_true, axis=(1, 2, 3), keepdims=True)
    y_true = y_true / (eps + sum_per_image)
    sum_per_image = mx.sum(y_pred, axis=(1, 2, 3), keepdims=True)
    y_pred = y_pred / (eps + sum_per_image)
    loss = y_true * mx.log(eps + y_true / (eps + y_pred))
    return mx.mean(mx.sum(loss, axis=(1, 2, 3)))
```

#### 7.6 Performance Characteristics on M3 Ultra

Based on comprehensive MLX benchmarks:

**MLX vs CUDA Performance:**
- The best performing M2 Ultra chip is faster than two Tesla V100 GPUs
- RTX4090 remains approximately 3x faster than M2 Ultra for raw compute
- CUDA V100 PCIe & NVLINK are only 23% and 34% faster than M3 Max with MLX

**MLX vs MPS (TensorFlow Metal):**
- MLX is usually much faster than MPS for most operations
- MLX can get only about 20%-30% benefit from FP16 optimization
- Large models train faster on MLX GPU compared to PyTorch MPS

#### 7.7 Existing VGG/CNN Implementations in MLX

The [mlx-image library](https://github.com/riccardomusmeci/mlx-image) provides pretrained image models, but **VGG16 is NOT currently available**. Available models include:
- ResNet (18, 34, 50, 101, 152)
- Vision Transformers (ViT)
- Swin Transformers

For VGG16, you would need to:
1. Convert existing TensorFlow/PyTorch VGG16 weights to MLX format
2. Use `mx.load()` with `.npz` or `.safetensors` files
3. Handle weight transposition from NCHW to NHWC format

#### 7.8 Migration Effort Assessment

| Task | Complexity | Notes |
|------|------------|-------|
| Rewrite MSINET class in MLX | Medium | Replace `tf.keras.Model` with `mlx.nn.Module` |
| Implement custom loss function | Low | Direct API translation |
| Create training loop with `value_and_grad` | Medium | Different pattern from TensorFlow |
| Convert VGG16 pretrained weights | Medium | Transpose conv kernels HWIO→OHWI |
| Adapt data loading pipeline | Medium | Replace `tf.data.Dataset` |
| Testing and debugging | High | New framework learning curve |

**MLX Training Loop Pattern:**
```python
loss_and_grad_fn = nn.value_and_grad(model, kld_loss)
optimizer = optim.Adam(learning_rate=1e-5)

for epoch in range(n_epochs):
    for images, targets in train_data:
        loss, grads = loss_and_grad_fn(model, images, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
```

#### 7.9 Trade-offs Summary

**Advantages of MLX:**
1. **Unified Memory Architecture**: No CPU-GPU data transfers, better memory efficiency
2. **Native Apple Silicon Optimization**: Designed specifically for M-series chips
3. **PyTorch-like API**: Familiar syntax for those coming from PyTorch
4. **Lazy Evaluation**: Optimized computation graphs
5. **Active Development**: Regular updates from Apple ML Research
6. **NHWC Native**: No need to handle format conversion at runtime

**Disadvantages of MLX:**
1. **Ecosystem Maturity**: Fewer pretrained models, tutorials, and community solutions
2. **No VGG16 Pretrained**: Would need to convert weights manually
3. **Limited Deployment Options**: Less production tooling than TensorFlow
4. **Apple-Only**: Code not portable to other platforms
5. **FP16 Performance**: Less optimized than NVIDIA Tensor Cores

#### 7.10 Recommendations

**For Inference Only (Production):**
Use TensorFlow-Metal - existing trained weights work directly with `channels_last` configuration.

**For Training and Development:**
Consider MLX if:
- You want maximum M3 Ultra performance
- You're comfortable with a less mature ecosystem
- You don't need cross-platform compatibility

Stick with TensorFlow-Metal if:
- You need to share code with non-Apple platforms
- You want access to extensive TensorFlow ecosystem
- Migration effort is a concern

---

## Code Architecture Summary

```
config.py
  |
  +-- PARAMS["device"] = "gpu" | "cpu" | "tpu"
          |
          v
model.py (MSINET.__init__)
  |
  +-- if device == "gpu":
  |       data_format = "channels_first" (NCHW)
  |
  +-- elif device in ("cpu", "tpu"):
          data_format = "channels_last" (NHWC)
          |
          v
      All Conv2D/MaxPooling2D layers configured with data_format
          |
          v
      Weight files saved with device suffix: model_{dataset}_{device}.weights.h5
```

---

## Code References

| File | Lines | Purpose |
|------|-------|---------|
| `config.py` | 13-18 | Device parameter configuration |
| `model.py` | 47-55 | Data format selection logic |
| `model.py` | 58-144 | Conv2D/MaxPooling layer definitions |
| `model.py` | 160-170 | Upsample format conversion |
| `model.py` | 283-294 | Weight save/load with device naming |
| `pyproject.toml` | 24-42 | Dependencies and optional GPU extras |
| `README.md` | 114 | TPU channels_last documentation |

---

## Web Sources

### TensorFlow Metal
- [Apple TensorFlow Metal Plugin](https://developer.apple.com/metal/tensorflow-plugin/)
- [tensorflow-metal PyPI](https://pypi.org/project/tensorflow-metal/)
- [tensorflow-metal Apple Developer Forums](https://developer.apple.com/forums/tags/tensorflow-metal/)
- [TensorFlow 2.19 on Apple Silicon M3 Guide](https://medium.com/@dr.saad.laouadi/installing-tensorflow-2-19-on-apple-silicon-m3-a-step-by-step-guide-b50fc3086a89)

### Apple MLX
- [Apple MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX 0.30.0 Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX Conv2d Documentation](https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.Conv2d.html)
- [MLX Layers Documentation](https://ml-explore.github.io/mlx/build/html/python/nn/layers.html)
- [MLX Optimizers Documentation](https://ml-explore.github.io/mlx/build/html/python/optimizers.html)
- [MLX Benchmarks - Towards Data Science](https://towardsdatascience.com/how-fast-is-mlx-a-comprehensive-benchmark-on-8-apple-silicon-chips-and-4-cuda-gpus-378a0ae356a0/)
- [PyTorch and MLX for Apple Silicon](https://towardsdatascience.com/pytorch-and-mlx-for-apple-silicon-4f35b9f60e39/)
- [mlx-image Library](https://github.com/riccardomusmeci/mlx-image)
- [MLX CNN Example](https://github.com/mikecvet/cnn)

### Hardware
- [Mac Studio M3 Ultra Specifications](https://www.apple.com/mac-studio/)
- [Mac Studio M3 Ultra AI Review](https://creativestrategies.com/mac-studio-m3-ultra-ai-workstation-review/)

---

## Critical Files for Implementation

- `config.py` - Device configuration that controls data format selection
- `model.py` - MSINET class with data format logic and all layer definitions
- `pyproject.toml` - Dependencies that need tensorflow-metal addition
- `main.py` - Training entry point that passes device configuration

---

## Open Questions

1. **Weight conversion:** Can existing `channels_first` weights be converted to `channels_last` format, or is retraining required?
2. **Performance benchmarks:** What is the actual performance difference between tensorflow-metal on M3 Ultra vs NVIDIA GPUs for this specific model?
3. **MLX vs TensorFlow-Metal benchmarks:** Direct comparison of training/inference speed for this specific CNN architecture on M3 Ultra would help guide the framework choice.
