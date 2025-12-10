# Download and setup the fixationadd1000 dataset for fine-tuning
setup-fixationadd1000:
    mkdir -p data/fixationadd1000
    gcloud storage cp gs://hmm-ml-models/fixation/add1000.tar.gz .
    tar -xzf add1000.tar.gz -C data/fixationadd1000
    mv data/fixationadd1000/fixation data/fixationadd1000/saliency
    rm add1000.tar.gz
    @echo "Dataset ready at data/fixationadd1000/"

# Train the base SALICON model
train-salicon:
    uv run python main.py train -d salicon

# Fine-tune on fixationadd1000 (requires SALICON model trained first)
train-fixationadd1000:
    uv run python main.py train -d fixationadd1000 -p data/

train: setup-fixationadd1000 train-salicon train-fixationadd1000

# Test model on a dataset
test dataset path:
    uv run python main.py test -d {{dataset}} -p {{path}}

# ============================================================================
# Vertex AI Training
# ============================================================================

# Configuration
REGION := "us-central1"
PROJECT := env("GOOGLE_CLOUD_PROJECT", "")
ARTIFACT_REGISTRY := REGION + "-docker.pkg.dev/" + PROJECT + "/saliency"
IMAGE_TAG := "latest"

# Build and push Docker image to Artifact Registry (GPU)
build-container:
    @echo "Building GPU training container..."
    docker build -t {{ARTIFACT_REGISTRY}}/msi-net-gpu:{{IMAGE_TAG}} -f Dockerfile .
    docker push {{ARTIFACT_REGISTRY}}/msi-net-gpu:{{IMAGE_TAG}}
    @echo "Image pushed to {{ARTIFACT_REGISTRY}}/msi-net-gpu:{{IMAGE_TAG}}"

# Build and push TPU Docker image to Artifact Registry
build-container-tpu:
    @echo "Building TPU training container..."
    docker build -t {{ARTIFACT_REGISTRY}}/msi-net-tpu:{{IMAGE_TAG}} -f Dockerfile.tpu .
    docker push {{ARTIFACT_REGISTRY}}/msi-net-tpu:{{IMAGE_TAG}}
    @echo "Image pushed to {{ARTIFACT_REGISTRY}}/msi-net-tpu:{{IMAGE_TAG}}"

# Submit Vertex AI training job for SALICON (GPU)
train-vertex-salicon:
    gcloud ai custom-jobs create \
        --region={{REGION}} \
        --display-name="msi-net-salicon-training" \
        --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri={{ARTIFACT_REGISTRY}}/msi-net-gpu:{{IMAGE_TAG}} \
        --args="train,-d,salicon" \
        --env-vars="GCS_OUTPUT_PATH=gs://hmm-ml-models/fixation/salicon/"

# Submit Vertex AI training job for fine-tuning on fixationadd1000 (GPU)
train-vertex-fixationadd1000:
    gcloud ai custom-jobs create \
        --region={{REGION}} \
        --display-name="msi-net-fixationadd1000-training" \
        --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri={{ARTIFACT_REGISTRY}}/msi-net-gpu:{{IMAGE_TAG}} \
        --args="train,-d,fixationadd1000,-p,gs://hmm-ml-models/fixation/add1000/" \
        --env-vars="GCS_OUTPUT_PATH=gs://hmm-ml-models/fixation/fixationadd1000/"

# Submit Vertex AI training job for SALICON (TPU v5e)
train-vertex-salicon-tpu:
    gcloud ai custom-jobs create \
        --region={{REGION}} \
        --display-name="msi-net-salicon-tpu-training" \
        --worker-pool-spec=machine-type=ct5lp-hightpu-1t,tpu-topology=1x1,replica-count=1,container-image-uri={{ARTIFACT_REGISTRY}}/msi-net-tpu:{{IMAGE_TAG}} \
        --args="train,-d,salicon" \
        --env-vars="GCS_OUTPUT_PATH=gs://hmm-ml-models/fixation/salicon/,PJRT_DEVICE=TPU"

# ============================================================================
# Model Registry
# ============================================================================

# Upload trained model to Vertex AI Model Registry
upload-model dataset="salicon" device="gpu":
    gcloud ai models upload \
        --region={{REGION}} \
        --display-name="msi-net-{{dataset}}" \
        --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-15:latest \
        --artifact-uri=gs://hmm-ml-models/fixation/{{dataset}}/ckpts/best/model_{{dataset}}_{{device}}/

# List models in registry
list-models:
    gcloud ai models list --region={{REGION}}
