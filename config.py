"""General training parameters that define the maximum number of
   training epochs, the batch size, and learning rate for the ADAM
   optimization method. To reproduce the results from the paper,
   these values should not be changed. The device can be either
   "cpu", "gpu", "tpu", or "metal", which then optimizes the model
   accordingly after training or uses the correct version for inference
   when testing.

   Device options:
     - "gpu": NVIDIA CUDA GPU (channels_first format, NCHW)
     - "cpu": CPU-only training/inference (channels_last format, NHWC)
     - "tpu": Google Cloud TPU (channels_last format, NHWC)
     - "metal": Apple Silicon GPU via tensorflow-metal (channels_last, NHWC)

   Note: TPU and Metal require channels_last data format (same as CPU).
   Weights trained with channels_last (cpu/tpu/metal) are cross-compatible.
   Weights trained with channels_first (gpu) are NOT compatible with
   channels_last inference.
"""

PARAMS = {
    "n_epochs": 10,
    "batch_size": 1,
    "learning_rate": 1e-5,
    "device": "gpu"
}

# GCS output path for model artifacts (set via environment variable or override here)
# Format: gs://bucket-name/path/to/models/{model}
# Example: gs://hmm-ml-models/fixation/{model}
import os
GCS_OUTPUT_PATH = os.environ.get("GCS_OUTPUT_PATH", None)

"""The predefined input image sizes for each of the 7 datasets.
   To reproduce the results from the paper, these values should
   not be changed. They must be divisible by 8 due to the model's
   downsampling operations. Furthermore, all pretrained models
   for download were trained on these image dimensions.
"""

DIMS = {
    "image_size_salicon": (240, 320),
    "image_size_mit1003": (360, 360),
    "image_size_cat2000": (216, 384),
    "image_size_dutomron": (360, 360),
    "image_size_pascals": (360, 360),
    "image_size_osie": (240, 320),
    "image_size_fiwi": (216, 384),
    "image_size_fixationadd1000": (360, 360)
}
