# Implementation Plan: Vertex AI GPU/TPU Training for Visual Saliency Model

This plan enables running the MSI-Net saliency model training on Google Cloud Vertex AI with GPU (and optionally TPU) accelerators. It covers containerization, GCS artifact storage integration, and the necessary code modifications.

## Context

**Research Document:** `docs/research/2025-12-10-vertex-ai-training.md`

**Relevant Files:**
- `config.py` - Device configuration and GCS output path (already has `GCS_OUTPUT_PATH` env var)
- `main.py` - Training entry point, path definitions (lines 35-56)
- `model.py` - Data format handling (lines 47-54), weight saving/export (lines 282-400)
- `utils.py` - History class saves to local filesystem (lines 69-100)
- `justfile` - Existing GCS integration pattern (line 4)
- `pyproject.toml` - Dependencies

**Key Findings:**
1. The codebase uses TensorFlow 2.15+ which is compatible with Vertex AI GPU containers
2. GPU mode uses `channels_first` data format - works with Vertex AI GPU
3. TPU requires `channels_last` format - needs explicit device option (config.py mentions TPU but model.py only handles cpu/gpu)
4. GCS_OUTPUT_PATH environment variable already exists but is not used anywhere
5. All output paths in main.py are hardcoded to local filesystem
6. Pre-built GPU container available: `us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest`

**Architecture Notes:**
- Current output structure uses local paths: `results/ckpts/best/`, `results/ckpts/latest/`, `results/history/`
- Model exports to SavedModel format compatible with Vertex AI Model Registry
- Existing GCS bucket pattern: `gs://hmm-ml-models/`

## Task list

- [x] **Task 1: Add TPU device support to model.py** (lines 47-54)
  - Add `elif config.PARAMS["device"] == "tpu":` condition
  - Use `channels_last` format (same as CPU) for TPU compatibility
  - File: `model.py`

- [x] **Task 2: Implement GCS output path support in main.py** (lines 35-56)
  - Modify `define_paths()` to check `config.GCS_OUTPUT_PATH`
  - When GCS path is set, use it as base for `results_path`
  - Keep local paths as fallback when env var is not set
  - TensorFlow can natively write to `gs://` paths
  - File: `main.py`

- [x] **Task 3: Update History class to support GCS paths** (lines 69-100)
  - TensorFlow's `tf.io.gfile` module can write to GCS transparently
  - Replace `os.makedirs` with `tf.io.gfile.makedirs`
  - Replace `open()` with `tf.io.gfile.GFile()`
  - Update plot saving to use `tf.io.gfile` or skip when on GCS
  - File: `utils.py`

- [x] **Task 4: Create Dockerfile for GPU training**
  - Base image: `us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-15.py310:latest`
  - Copy source files and install dependencies from `pyproject.toml`
  - Set entrypoint to `python main.py`
  - Create at: `Dockerfile`

- [x] **Task 5: Create Dockerfile.tpu for TPU training** (optional)
  - Base image: Python 3.10 slim
  - Install TensorFlow TPU wheel from `gs://cloud-tpu-tpuvm-artifacts/tensorflow/tf-2.15.0/`
  - Download libtpu.so to `/lib/`
  - Set `PJRT_DEVICE=TPU` environment variable
  - Create at: `Dockerfile.tpu`

- [x] **Task 6: Add justfile targets for Vertex AI training**
  - `build-container`: Build and push Docker image to Artifact Registry
  - `train-vertex-salicon`: Submit Vertex AI training job for SALICON
  - `train-vertex-fixationadd1000`: Submit Vertex AI training job for fine-tuning
  - Use `gcloud ai custom-jobs create` command
  - File: `justfile`

- [x] **Task 7: Add justfile target for model upload to Vertex AI Model Registry**
  - `upload-model`: Upload trained SavedModel to Model Registry
  - Use `gcloud ai models upload` command
  - Configure artifact URI to GCS SavedModel path
  - File: `justfile`

- [x] **Task 8: Update README with Vertex AI training instructions**
  - Add section on cloud training setup
  - Document environment variables (GCS_OUTPUT_PATH)
  - Include example commands for submitting training jobs
  - File: `README.md`

- [x] **Task 9: Create vertex-ai-config.yaml for training job specification** (optional)
  - Define machine type, GPU count, container image
  - Set environment variables for GCS paths
  - Allow parameterization for different datasets
  - Create at: `vertex-ai-config.yaml`

## Refs

- Research document: `docs/research/2025-12-10-vertex-ai-training.md`
- [Prebuilt containers for Vertex AI serverless training](https://docs.cloud.google.com/vertex-ai/docs/training/pre-built-containers)
- [Create a custom container image for training](https://docs.cloud.google.com/vertex-ai/docs/training/create-custom-container)
- [Training with TPU accelerators](https://docs.cloud.google.com/vertex-ai/docs/training/training-with-tpu-vm)
- [TensorFlow integration with Vertex AI](https://docs.cloud.google.com/vertex-ai/docs/start/tensorflow)
