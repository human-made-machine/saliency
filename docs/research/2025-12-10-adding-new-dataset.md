---
date: 2025-12-10T00:00:00+00:00
researcher: Claude
topic: "Adding a New Dataset for Fine-Tuning"
tags: [research, codebase, dataset, fine-tuning, data-pipeline]
last_updated: 2025-12-10
last_updated_by: Claude
---

# Research: Adding a New Dataset for Fine-Tuning

## Research Question

I want to add a new dataset to fine tune the model after the base training. It is in the file `fixation_add1000.tar.gz`. Investigate how to do this.

## Summary

The codebase has a well-defined pattern for adding datasets. Each dataset is a Python class in `data.py` with specific attributes and a `load_data()` method. To add the new `fixation_add1000` dataset for fine-tuning, modifications are needed in 4 files: `data.py`, `config.py`, `main.py`, and `model.py`.

---

## Detailed Findings

### 1. Current Dataset Architecture

#### 1.1 Dataset Classes in `data.py`

The codebase defines dataset classes in `/home/josh/p/src/github.com/jfyne/saliency/data.py`. Each dataset is a Python class with the following pattern:

**Existing datasets (lines 11-437):**
- `SALICON` (lines 11-65) - Base training dataset with 10,000 train / 5,000 validation images
- `MIT1003` (lines 68-125) - 803 train / 200 validation images
- `CAT2000` (lines 128-196) - 1,600 train / 400 validation images
- `DUTOMRON` (lines 199-257) - 4,168 train / 1,000 validation images
- `PASCALS` (lines 260-318) - 650 train / 200 validation images
- `OSIE` (lines 321-377) - 500 train / 200 validation images
- `FIWI` (lines 380-436) - 99 train / 50 validation images

#### 1.2 Common Dataset Class Structure

Each dataset class follows a consistent pattern:

```python
class DATASETNAME:
    """Docstring with dataset description."""

    n_train = X  # Number of training instances
    n_valid = Y  # Number of validation instances

    def __init__(self, data_path):
        self._target_size = config.DIMS["image_size_datasetname"]
        self._dir_stimuli = data_path + "stimuli"
        self._dir_saliency = data_path + "saliency"

        if not os.path.exists(data_path):
            # Download dataset if not present
            parent_path = os.path.dirname(data_path[:-1])
            parent_path = os.path.join(parent_path, "")
            download.download_datasetname(parent_path)

    def load_data(self):
        # Load file lists, split into train/valid, return datasets
        ...
        return (train_set, valid_set)
```

#### 1.3 Dataset Directory Structures

**SALICON structure** (uses train/val subdirectories):
```
data/salicon/
  stimuli/
    train/
      COCO_train2014_000000000009.jpg
      ...
    val/
      ...
  saliency/
    train/
      COCO_train2014_000000000009.png
      ...
    val/
      ...
```

**Other datasets** (use flat directory with random train/valid split):
```
data/mit1003/
  stimuli/
    image1.jpeg
    image2.jpeg
    ...
  saliency/
    image1.png
    image2.png
    ...
```

#### 1.4 Your Archive Structure (`fixation_add1000.tar.gz`)

Based on standard naming conventions, the archive likely contains:
- **Top-level directories:** `stimuli/` and `fixation/`
- **Stimuli format:** `stimuli/499.jpg`, `stimuli/727.jpg`, etc. (1,000 files)
- **Fixation format:** `fixation/54.png`, `fixation/403.png`, etc. (1,000 files)

**Key observation:** The archive has a `fixation/` directory, but the codebase expects a `saliency/` directory. Fixation maps typically need to be converted to blurred saliency maps for training (as seen in the download functions for DUTOMRON, OSIE, FIWI).

---

### 2. Configuration Requirements

#### 2.1 Image Size Configuration in `config.py`

Location: `/home/josh/p/src/github.com/jfyne/saliency/config.py` (lines 23-31)

```python
DIMS = {
    "image_size_salicon": (240, 320),
    "image_size_mit1003": (360, 360),
    "image_size_cat2000": (216, 384),
    "image_size_dutomron": (360, 360),
    "image_size_pascals": (360, 360),
    "image_size_osie": (240, 320),
    "image_size_fiwi": (216, 384)
}
```

Each dataset requires an entry in this dictionary. Image sizes must be **divisible by 8** due to the model's downsampling operations.

#### 2.2 Dataset List in `main.py`

Location: `/home/josh/p/src/github.com/jfyne/saliency/main.py` (lines 215-216)

```python
datasets_list = ["salicon", "mit1003", "cat2000",
                 "dutomron", "pascals", "osie", "fiwi"]
```

New datasets must be added to this list to be selectable via command line arguments.

---

### 3. Training/Fine-Tuning Flow

#### 3.1 Dataset Loading Process

From `main.py` (line 74):
```python
train_ds, valid_ds = data.get_dataset_iterator("train", dataset, paths["data"])
```

From `data.py` (lines 463-486), the `get_dataset_iterator` function:
1. Uses reflection to find the class by name: `getattr(current_module, class_name.upper())`
2. Instantiates the dataset class with the data path
3. Calls `load_data()` method to get train and validation datasets

#### 3.2 Weight Restoration for Fine-Tuning

From `model.py` (lines 295-332), the `restore` method handles fine-tuning:

```python
def restore(self, dataset, paths, device):
    ...
    elif dataset in ("mit1003", "cat2000", "dutomron",
                     "pascals", "osie", "fiwi"):
        if os.path.isfile(paths["best"] + salicon_name + weights_ext):
            self.load_weights(paths["best"] + salicon_name + weights_ext)
            print(">> Restored weights from SALICON checkpoint")
        else:
            raise FileNotFoundError("Train model on SALICON first")
```

**Key requirement:** For fine-tuning, the new dataset name must be added to this list (line 315-316) to enable loading pre-trained SALICON weights. Otherwise, it will try to start from VGG16 weights or random initialization.

---

### 4. File Naming Consistency Check

From `data.py` (lines 767-784), `_check_consistency` verifies:
- Stimuli and saliency files have matching base names
- Suffixes like `_fixMap` and `_fixPts` are stripped for comparison

```python
file_names = [entry.replace("_fixMap", "") for entry in file_names]
file_names = [entry.replace("_fixPts", "") for entry in file_names]
```

Your archive files should use matching numeric naming (`499.jpg` / `499.png`) between stimuli and saliency directories.

---

## Code References

| File | Location | Purpose |
|------|----------|---------|
| `data.py` | Lines 11-437 | Existing dataset class definitions |
| `data.py` | Lines 463-486 | `get_dataset_iterator()` function |
| `data.py` | Lines 767-784 | `_check_consistency()` function |
| `config.py` | Lines 23-31 | `DIMS` dictionary with image sizes |
| `main.py` | Lines 215-216 | `datasets_list` for CLI parsing |
| `model.py` | Lines 295-332 | `restore()` method for weight loading |
| `model.py` | Lines 315-316 | Fine-tuning dataset list |
| `download.py` | Lines 221-227 | Example of fixation-to-saliency conversion |

---

## Files to Modify

| File | Location | Modification Required |
|------|----------|----------------------|
| `data.py` | Lines 437+ | Add new dataset class (e.g., `FIXATIONADD1000`) |
| `config.py` | Lines 23-31 | Add `"image_size_fixationadd1000": (height, width)` |
| `main.py` | Lines 215-216 | Add `"fixationadd1000"` to `datasets_list` |
| `model.py` | Lines 315-316 | Add `"fixationadd1000"` to fine-tuning dataset list |
| `download.py` (optional) | Lines 440+ | Add download function if hosting archive remotely |

---

## Directory Structure Preparation

Before training, the archive needs to be extracted and potentially restructured:

**Expected structure for flat dataset (like MIT1003):**
```
data/fixationadd1000/
  stimuli/
    1.jpg
    2.jpg
    ...
  saliency/      <-- Note: renamed from "fixation"
    1.png
    2.png
    ...
```

**Optional:** If fixation maps are binary point maps (not blurred), they may need conversion to saliency maps using Gaussian filtering, as done in `download.py` for DUTOMRON (lines 221-227):
```python
saliency_map = gaussian_filter(fixations_map, 16)
saliency_map_normalized = (saliency_map / saliency_map.max() * 255).astype(np.uint8)
```

---

## Dataset Class Pattern to Follow

Based on `MIT1003` class (simplest pattern for a flat directory structure):

```python
class FIXATIONADD1000:
    """Dataset class for the fixation_add1000 dataset."""

    n_train = 800   # Adjust based on desired train/valid split
    n_valid = 200

    def __init__(self, data_path):
        self._target_size = config.DIMS["image_size_fixationadd1000"]

        self._dir_stimuli = data_path + "stimuli"
        self._dir_saliency = data_path + "saliency"

        # Optional: Add download logic if needed

    def load_data(self):
        list_x = _get_file_list(self._dir_stimuli)
        list_y = _get_file_list(self._dir_saliency)

        _check_consistency(zip(list_x, list_y), 1000)

        indices = _get_random_indices(1000)
        excerpt = indices[:self.n_train]

        train_list_x = [list_x[idx] for idx in excerpt]
        train_list_y = [list_y[idx] for idx in excerpt]

        train_set = _fetch_dataset((train_list_x, train_list_y),
                                   self._target_size, True)

        excerpt = indices[self.n_train:]

        valid_list_x = [list_x[idx] for idx in excerpt]
        valid_list_y = [list_y[idx] for idx in excerpt]

        valid_set = _fetch_dataset((valid_list_x, valid_list_y),
                                   self._target_size, False)

        return (train_set, valid_set)
```

---

## Training Command After Implementation

Once implemented, fine-tuning would be executed with:

```bash
# First, ensure SALICON model is trained (creates checkpoint in results/ckpts/best/)
uv run python main.py train -d salicon

# Then fine-tune on the new dataset
uv run python main.py train -d fixationadd1000 -p data/
```

The model will:
1. Look for existing checkpoint at `results/ckpts/latest/model_fixationadd1000_gpu.weights.h5`
2. If not found, load SALICON weights from `results/ckpts/best/model_salicon_gpu.weights.h5`
3. Train for 10 epochs (as configured in `config.py`)
4. Save checkpoints to `results/ckpts/latest/` and best model to `results/ckpts/best/`

---

## Open Questions

1. **What is the exact structure inside `fixation_add1000.tar.gz`?** Need to verify directory names and file formats.
2. **Are the fixation maps already blurred saliency maps, or raw fixation points?** If raw, they need Gaussian filtering.
3. **What image dimensions should be used?** Need to determine appropriate `(height, width)` for the config.
4. **What train/valid split ratio is desired?** Currently using 800/200 as an example.
