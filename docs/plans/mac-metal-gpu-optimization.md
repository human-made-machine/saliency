# Implementation Plan

Based on the research document at `docs/research/2025-12-11-mac-m3-ultra-gpu-optimization.md`, this plan enables running the MSI-Net model with `channels_last` data format optimized for Apple Metal GPU acceleration. However, a critical clarification is needed: **tensorflow-metal is macOS-only and cannot be used on Linux**. The `channels_last` format is required for Metal but also happens to be the format used for CPU/TPU training, making weights trained this way portable across platforms.

## Context

**Research Document:** `docs/research/2025-12-11-mac-m3-ultra-gpu-optimization.md`

**Critical Finding:** tensorflow-metal is exclusively designed for macOS on Apple Silicon. It cannot be installed or used on Linux. If the goal is Linux deployment, you need to:
1. Train on Mac with Metal using `channels_last` format
2. Deploy the trained weights on Linux using CPU or NVIDIA GPU (with `channels_last` configuration)

**Relevant Files:**
- `config.py` (lines 13-18) - Device configuration
- `model.py` (lines 47-55) - Data format selection logic
- `pyproject.toml` (lines 35-38) - Optional dependencies
- `main.py` - Training entry point

**Key Architecture Notes:**
1. The codebase already supports `channels_last` via `device: "cpu"` or `device: "tpu"` (see model.py lines 51-55)
2. Metal GPU requires `channels_last` (NHWC) format - same as existing CPU/TPU paths
3. Weight files are saved with device suffix: `model_{dataset}_{device}.weights.h5`
4. Weights trained with `channels_first` (GPU/CUDA) are NOT compatible with `channels_last` inference

**Compatibility Matrix:**

| Device Setting | Data Format | Platform | GPU Acceleration |
|----------------|-------------|----------|------------------|
| `gpu` | channels_first (NCHW) | Linux/Windows with NVIDIA | CUDA |
| `cpu` | channels_last (NHWC) | Any | None |
| `tpu` | channels_last (NHWC) | GCP | TPU |
| `metal` (new) | channels_last (NHWC) | macOS Apple Silicon | Metal |

## Task list

- [x] **Task 1: Add "metal" device option to config.py** (lines 13-18)
  - Add documentation comment explaining Metal GPU requirements
  - The default device value remains "gpu" for CUDA compatibility
  - Users can switch to "metal" for Apple Silicon training

- [x] **Task 2: Update model.py device handling** (lines 47-55)
  - Add `"metal"` to the condition that uses `channels_last` format
  - Modify: `elif config.PARAMS["device"] in ("cpu", "tpu", "metal"):`
  - This enables Metal GPU to use the correct NHWC data format

- [x] **Task 3: Add tensorflow-metal optional dependency to pyproject.toml** (lines 35-38)
  - Create new optional dependency group: `metal = ["tensorflow-metal>=1.0.0"]`
  - Note: This package only installs on macOS ARM64

- [x] **Task 4: Add Metal GPU detection and logging in main.py**
  - Add optional GPU detection at startup when device is "metal"
  - Log available Metal devices using `tf.config.list_physical_devices('GPU')`
  - Warn if no Metal GPU detected when device="metal"

- [x] **Task 5: Update model.py restore/save to handle "metal" device naming**
  - Ensure weight files follow naming: `model_{dataset}_metal.weights.h5`
  - Consider adding weight compatibility note (metal weights compatible with cpu/tpu)

- [x] **Task 6: Update README.md with Apple Silicon / Metal GPU instructions**
  - Add new section "Apple Silicon (Mac) Training"
  - Document installation: `uv sync --extra metal` or `pip install tensorflow-metal`
  - Explain device configuration change
  - Note that `channels_last` weights are cross-platform compatible

- [x] **Task 7: Add justfile targets for local Mac training**
  - `train-salicon-metal`: Train SALICON with device=metal
  - `train-fixationadd1000-metal`: Fine-tune with device=metal
  - These would modify config or pass device override

- [x] **Task 8: Document weight conversion between formats** (optional, if needed)
  - Research document notes that `channels_first` weights cannot be directly used with `channels_last`
  - Document whether conversion is possible or if retraining is required
  - Add to research doc or create knowledge doc at: `docs/knowledge/weight-format-compatibility.md`

## Implementation Notes

### Why "metal" as a separate device option?

1. **Clarity**: Explicitly indicates Apple Metal GPU backend
2. **Weight Naming**: Weights saved as `model_salicon_metal.weights.h5` are clearly identified
3. **Future-proofing**: Allows Metal-specific optimizations if needed

### Alternative: Reuse "cpu" device

The existing `device: "cpu"` already uses `channels_last` format which is Metal-compatible. Users could simply:
1. Set `device: "cpu"` in config.py
2. Install tensorflow-metal
3. TensorFlow automatically uses Metal GPU when available

However, this approach:
- Produces weights named `model_salicon_cpu.weights.h5` (misleading)
- Does not clearly indicate Metal GPU usage in logs
- May confuse future users about which device was used

### Linux Deployment Clarification

If the goal is to train on Mac and deploy on Linux:
1. Train with `device: "metal"` (or "cpu"/"tpu") - all use `channels_last`
2. Copy weights to Linux machine
3. On Linux, use `device: "cpu"` configuration for inference
4. Weights are compatible because both use `channels_last` format

The CUDA `device: "gpu"` option on Linux uses `channels_first` and requires different weights.

## Refs

- Research document: `docs/research/2025-12-11-mac-m3-ultra-gpu-optimization.md`
- [tensorflow-metal PyPI](https://pypi.org/project/tensorflow-metal/)
- [Apple TensorFlow Metal Plugin](https://developer.apple.com/metal/tensorflow-plugin/)
- [TensorFlow GPU Plugins](https://www.tensorflow.org/install/gpu_plugins)
