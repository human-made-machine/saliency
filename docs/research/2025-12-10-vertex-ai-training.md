---
date: 2025-12-10T12:00:00Z
researcher: Claude
topic: "Vertex AI GPU/TPU Training for Visual Saliency Model"
tags: [research, vertex-ai, gpu, tpu, tensorflow, training, gcs]
last_updated: 2025-12-10
last_updated_by: Claude
---

# Research: Vertex AI GPU/TPU Training for Visual Saliency Model

## Research Question

I need to run the base and fine tuning training on a GPU. I want to use Google Cloud Vertex AI to do this.
- Can this code run on a TPU?
- What are my best options for doing this and storing the model artifact?

## Summary

The MSI-Net saliency model in this codebase is built with TensorFlow 2.15+ and uses a device-dependent data format configuration. **The code can run on TPU with modifications** - the current GPU mode uses `channels_first` data format which is incompatible with TPU. TPU requires `channels_last` format, which the codebase already supports via its CPU mode. For Vertex AI deployment, **GPU training is the most straightforward option**, while TPU training requires switching to `channels_last` format and using specialized TPU containers.

## Detailed Findings

### 1. Current Training Infrastructure

#### 1.1 Framework and Dependencies

Location: `pyproject.toml:24-33`

```python
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

The project uses **TensorFlow 2.15+** which is compatible with both GPU and TPU v5e on Vertex AI.

#### 1.2 Device Configuration and Data Format

Location: `config.py:9-14`

```python
PARAMS = {
    "n_epochs": 10,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "device": "gpu"
}
```

Location: `model.py:47-54`

```python
if config.PARAMS["device"] == "gpu":
    self._data_format = "channels_first"
    self._channel_axis = 1
    self._dims_axis = (2, 3)
elif config.PARAMS["device"] == "cpu":
    self._data_format = "channels_last"
    self._channel_axis = 3
    self._dims_axis = (1, 2)
```

**Key finding:** The model uses `channels_first` (NCHW) for GPU and `channels_last` (NHWC) for CPU. This is critical for TPU compatibility.

#### 1.3 Training Flow

Location: `main.py:61-139`

The training process:
1. Loads dataset iterator via `data.get_dataset_iterator()`
2. Creates MSI-Net model instance
3. Restores weights from checkpoints or VGG16 pretrained weights
4. Runs training loop with gradient tape
5. Saves weights and exports SavedModel format

#### 1.4 Model Export Format

Location: `model.py:372-400`

```python
def export_saved_model(self, dataset, path, device):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3],
                                                 dtype=tf.float32, name="input")])
    def serving_fn(input_tensor):
        output = self(input_tensor, training=False)
        return {"output": output}

    tf.saved_model.save(
        self,
        model_path,
        signatures={"serving_default": serving_fn}
    )
```

The model exports in TensorFlow SavedModel format with `serving_default` signature, which is compatible with Vertex AI Model Registry.

### 2. TPU Compatibility Analysis

#### 2.1 Data Format Requirement

**TPU requires `channels_last` (NHWC) data format.** The current codebase:
- **GPU mode**: Uses `channels_first` - **NOT compatible with TPU**
- **CPU mode**: Uses `channels_last` - **Compatible with TPU**

#### 2.2 Running on TPU

To run this code on TPU, you would need to set `device: "cpu"` in config.py (or add a new `"tpu"` device option). This enables the `channels_last` data format that TPU requires.

From `model.py:159-167`:
```python
def _upsample(self, stack, target_shape, factor):
    if self._data_format == "channels_first":
        stack = tf.transpose(stack, (0, 2, 3, 1))
    # ... resize operation ...
    if self._data_format == "channels_first":
        stack = tf.transpose(stack, (0, 3, 1, 2))
```

The model already handles format conversion internally for resize operations, so the architecture is designed to work with both formats.

#### 2.3 TPU Version Support

- **TPU v5e**: Supports TensorFlow 2.15+ (compatible)
- **TPU v2/v3**: Supported but older generation
- **TPU v6e**: Requires TensorFlow 2.18+ (nightly) - NOT compatible with current codebase

### 3. Vertex AI Training Options

#### 3.1 GPU Training (Recommended for This Codebase)

**Advantages:**
- No code changes required
- Uses optimized `channels_first` format for GPU
- Supports NVIDIA T4, V100, A100, H100 GPUs

**Configuration options:**

| Machine Type | GPU Type | GPU Count | Use Case |
|-------------|----------|-----------|----------|
| n1-standard-4 | NVIDIA_TESLA_T4 | 1 | Cost-effective training |
| n1-standard-8 | NVIDIA_TESLA_V100 | 1 | Balanced performance |
| a2-highgpu-1g | NVIDIA_A100 | 1 | High performance |

#### 3.2 TPU Training (Requires Modification)

**Requirements:**
1. Change `config.py` device setting or add TPU support
2. Use TPU-specific TensorFlow wheel from Google Cloud Storage
3. Configure PJRT runtime environment variables

**TPU v5e machine types:**
- `ct5lp-hightpu-1t` (1 TPU chip)
- `ct5lp-hightpu-4t` (4 TPU chips)
- `ct5lp-hightpu-8t` (8 TPU chips)

#### 3.3 Prebuilt vs Custom Containers

**Option A: Prebuilt Container (GPU only)**

- TensorFlow 2.15 GPU containers available
- Use image URI: `us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest`

**Option B: Custom Container (GPU or TPU)**

Required for TPU training. Example Dockerfile structure:

```dockerfile
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-15
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "main.py", "train", "-d", "salicon"]
```

### 4. Model Artifact Storage

#### 4.1 GCS Bucket Best Practices

1. **Region alignment**: Store artifacts in same region as Vertex AI endpoint
2. **Bucket structure**:
   ```
   gs://your-bucket/
     models/
       salicon/
         v1/
           model_salicon_gpu.weights.h5
           saved_model/
         v2/
           ...
       fixationadd1000/
         v1/
           ...
     datasets/
       salicon/
       fixationadd1000/
   ```

3. **Use Anywhere Cache** for high-throughput model loading (up to 2.5 TB/s)

#### 4.2 Current Output Locations

From `main.py:35-44`:

```python
results_path = current_path + "/results/"
weights_path = current_path + "/weights/"

history_path = results_path + "history/"
images_path = results_path + "images/"
ckpts_path = results_path + "ckpts/"

best_path = ckpts_path + "best/"
latest_path = ckpts_path + "latest/"
```

**Artifacts to store:**
- `results/ckpts/best/model_{dataset}_{device}.weights.h5` - Best model weights
- `results/ckpts/best/model_{dataset}_{device}/` - SavedModel directory
- `results/history/train_{dataset}_{device}.txt` - Training history
- `results/history/valid_{dataset}_{device}.txt` - Validation history

#### 4.3 Vertex AI Model Registry Integration

The exported SavedModel format is directly compatible with Vertex AI Model Registry:

```bash
# Upload model to Vertex AI
gcloud ai models upload \
  --region=us-central1 \
  --display-name=msi-net-salicon \
  --artifact-uri=gs://your-bucket/models/salicon/v1/saved_model/ \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-15:latest
```

### 5. Existing Cloud Integration

Location: `justfile:1-8`

```bash
# Download and setup the fixationadd1000 dataset for fine-tuning
setup-fixationadd1000:
    mkdir -p data/fixationadd1000
    gcloud storage cp gs://hmm-ml-models/fixation/add1000.tar.gz .
    tar -xzf add1000.tar.gz -C data/fixationadd1000
    mv data/fixationadd1000/fixation data/fixationadd1000/saliency
    rm add1000.tar.gz
```

The project already uses GCS for dataset storage (`gs://hmm-ml-models/`), indicating existing Google Cloud infrastructure.

## Code References

| File | Location | Purpose |
|------|----------|---------|
| `config.py` | Lines 9-14 | Device and training parameters |
| `model.py` | Lines 47-54 | Data format selection (channels_first/last) |
| `model.py` | Lines 372-400 | SavedModel export |
| `main.py` | Lines 35-44 | Output path configuration |
| `main.py` | Lines 61-139 | Training loop |
| `justfile` | Lines 1-8 | GCS integration example |

## Architecture Documentation

The codebase follows a modular architecture:
- `config.py` - Central configuration with device selection
- `model.py` - MSINET class implementing the visual saliency model with VGG16 backbone
- `data.py` - Dataset loading and preprocessing
- `main.py` - CLI entry point for training, testing, and export operations

The device configuration in `config.py` controls the data format used throughout the model, making it the single point of change for GPU/CPU/TPU compatibility.

## Related Research

No prior research documents found in `docs/research/`.

## Open Questions

1. Should a dedicated `"tpu"` device option be added to `config.py` rather than reusing `"cpu"` mode?
2. What is the preferred GCS bucket naming convention for this project's model artifacts?
3. Should the training script be modified to support GCS paths directly for output?
