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
