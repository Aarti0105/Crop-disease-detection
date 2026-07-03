# =============================================================================
# data_loader.py  —  Data Loading and Preprocessing Pipeline
# =============================================================================
#
# PURPOSE:
#   This file handles everything related to preparing data BEFORE it goes
#   into a model. It is completely separate from model building and training
#   so you can swap datasets without touching model code.
#
# WHAT THIS FILE DOES (in order):
#   1. Load images and labels from .npy files
#   2. Normalise pixel values from [0–255] → [0.0–1.0]
#   3. Encode string labels ("Tomato_healthy") → integers → one-hot vectors
#   4. Split data into Train / Validation / Test sets
#   5. Return everything neatly packaged
#
# CONCEPT — Why normalise?
#   Neural networks work with small floating-point numbers.
#   Raw pixel values (0–255) are too large and make training unstable.
#   Dividing by 255 brings every value into [0.0, 1.0] — the model
#   converges faster and more reliably.
#
# CONCEPT — What is one-hot encoding?
#   Labels like "Tomato_Early_blight" must be converted to numbers.
#   Step 1: "Tomato_Early_blight" → 2  (integer label)
#   Step 2: 2 → [0, 0, 1, 0, 0, ...]  (one-hot vector, length = num_classes)
#   The model outputs a vector like this and we compare them.
# =============================================================================


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

 
# =============================================================================
# CLASS: LeafDataset
# =============================================================================
class LeafDataset(Dataset):
    """
    Custom PyTorch Dataset for the PlantVillage leaf images.
 
    CONCEPT — Why a Dataset class?
        PyTorch's DataLoader needs to know:
          (a) how many samples there are  →  __len__
          (b) how to fetch one sample     →  __getitem__
        We define both methods here.
        The DataLoader then calls __getitem__ repeatedly to build batches.
 
    Parameters
    ----------
    images    : np.ndarray  shape (N, 224, 224, 3)  uint8
    labels    : np.ndarray  shape (N,)              int  (encoded class indices)
    transform : torchvision.transforms  applied to each image when fetched
    """
 
    def __init__(self, images, labels, transform=None):
        # Store as numpy arrays — DataLoader will convert to tensors
        self.images    = images
        self.labels    = labels
        self.transform = transform
 
    def __len__(self):
        # Returns the total number of samples — DataLoader uses this
        return len(self.images)
 
    def __getitem__(self, idx):
        """
        Fetch one image-label pair by index.
 
        The DataLoader calls this for each item in a batch.
        transform converts the numpy image to a normalised PyTorch tensor.
        """
        # Get one image as a numpy array (H, W, C) uint8 [0-255]
        image = self.images[idx]
 
        # Get corresponding label as an integer
        label = self.labels[idx]
 
        # Apply transforms (ToTensor + Normalize)
        # ToTensor:   (H,W,C) uint8 [0,255]  →  (C,H,W) float32 [0,1]
        # Normalize:  applies ImageNet mean/std per channel
        if self.transform:
            image = self.transform(image)
 
        # Return as PyTorch tensor and Python int
        return image, torch.tensor(label, dtype=torch.long)
 
 
# =============================================================================
# FUNCTION: get_transforms
# =============================================================================
def get_transforms(config):
    """
    Build torchvision transform pipelines for train and validation/test sets.
 
    WHY DIFFERENT TRANSFORMS FOR TRAIN AND VAL/TEST?
        Train:  we apply random flips to artificially expand the dataset
                (a flipped leaf still has the same disease)
        Val/Test: no random transforms — we want deterministic results
 
    Returns
    -------
    train_transform : transforms.Compose
    eval_transform  : transforms.Compose  (used for val and test)
    """
 
    mean = config["IMAGENET_MEAN"]
    std  = config["IMAGENET_STD"]
 
    # Training transforms — includes random flip for slight augmentation
    train_transform = transforms.Compose([
        # ToTensor converts numpy (H,W,C) uint8 → torch (C,H,W) float32 [0,1]
        transforms.ToTensor(),
 
        # Random horizontal flip — leaves look the same flipped
        # p=0.5 means 50% chance of flipping each image each epoch
        transforms.RandomHorizontalFlip(p=0.5),
 
        # Normalise using ImageNet stats (REQUIRED for pretrained torchvision models)
        # Formula: output = (input - mean) / std   applied per channel
        transforms.Normalize(mean=mean, std=std),
    ])
 
    # Validation / Test transforms — no random operations
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
 
    return train_transform, eval_transform
 
 
# =============================================================================
# FUNCTION: prepare_data
# =============================================================================
def prepare_data(images_path, labels_path, config):
    """
    Master function — runs the full data preparation pipeline.
 
    Steps:
        1. Load images and labels from .npy files
        2. Encode string labels → integer indices
        3. Split into train / val / test  (64% / 16% / 20%)
        4. Wrap each split in a LeafDataset
        5. Wrap each dataset in a DataLoader (handles batching + shuffling)
 
    Parameters
    ----------
    images_path : str    path to images .npy file
    labels_path : str    path to labels .npy file
    config      : dict   from config.py
 
    Returns
    -------
    loaders     : dict   {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    test_labels : np.ndarray  integer labels for the test set (for sklearn metrics)
    class_names : np.ndarray  ordered class name strings
    encoder     : LabelEncoder
    """
 
    print()
    print("=" * 55)
    print("  DATA PREPARATION PIPELINE  (PyTorch)")
    print("=" * 55)
    print()
 
    # ── Step 1: Load from disk ─────────────────────────────────────────────
    print("  Step 1 — Loading .npy files from disk...")
    images = np.load(images_path)
    labels = np.load(labels_path, allow_pickle=True)
 
    print(f"  Images : {images.shape}  dtype={images.dtype}")
    print(f"  Labels : {labels.shape}  (string class names)")
    print()
 
    # ── Step 2: Encode string labels → integers ────────────────────────────
    # PyTorch CrossEntropyLoss needs integer class indices, NOT one-hot vectors
    # LabelEncoder maps "Tomato_healthy" → 3, "Potato___Late_blight" → 7, etc.
    print("  Step 2 — Encoding labels (string → integer)...")
    encoder     = LabelEncoder()
    labels_int  = encoder.fit_transform(labels)   # shape (N,) of int
    class_names = encoder.classes_
 
    print(f"  {len(class_names)} classes detected")
    print(f"  Example: '{class_names[0]}' → 0")
    print()
 
    # ── Step 3: Train / Val / Test split ──────────────────────────────────
    # Same split as before — 64% train / 16% val / 20% test
    # stratify keeps class balance in every split
    print("  Step 3 — Splitting dataset...")
 
    X_tv, X_test, y_tv, y_test = train_test_split(
        images, labels_int,
        test_size    = config["TEST_SIZE"],
        random_state = config["RANDOM_STATE"],
        stratify     = labels_int,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size    = config["VAL_SIZE"],
        random_state = config["RANDOM_STATE"],
        stratify     = y_tv,
    )
 
    print(f"  Train : {X_train.shape[0]} images")
    print(f"  Val   : {X_val.shape[0]}   images")
    print(f"  Test  : {X_test.shape[0]}  images")
    print()
 
    # ── Step 4: Build transforms ───────────────────────────────────────────
    train_tf, eval_tf = get_transforms(config)
 
    # ── Step 5: Wrap in Dataset ────────────────────────────────────────────
    train_dataset = LeafDataset(X_train, y_train, transform=train_tf)
    val_dataset   = LeafDataset(X_val,   y_val,   transform=eval_tf)
    test_dataset  = LeafDataset(X_test,  y_test,  transform=eval_tf)
 
    # ── Step 6: Wrap in DataLoader ─────────────────────────────────────────
    # DataLoader handles:
    #   batch_size → groups samples into batches
    #   shuffle    → randomises order each epoch (only for training)
    #   num_workers→ parallel CPU threads for loading (0 = main thread)
    #
    # IMPORTANT: shuffle=True only for training!
    # Val and test must NOT be shuffled so results are deterministic.
    loaders = {
        "train" : DataLoader(
            train_dataset,
            batch_size  = config["BATCH_SIZE"],
            shuffle = True,
            num_workers = config["NUM_WORKERS"],
        ),
        "val" : DataLoader(
            val_dataset,
            batch_size  = config["BATCH_SIZE"],
            shuffle = False,
            num_workers = config["NUM_WORKERS"],
        ),
        "test" : DataLoader(
            test_dataset,
            batch_size  = config["BATCH_SIZE"],
            shuffle = False,
            num_workers = config["NUM_WORKERS"],
        ),
    }
 
    print("  DataLoaders ready")
    print(f"  Train batches : {len(loaders['train'])}")
    print(f"  Val   batches : {len(loaders['val'])}")
    print(f"  Test  batches : {len(loaders['test'])}")
    print()
 
    # y_test is kept as numpy for sklearn classification_report later
    return loaders, y_test, class_names, encoder
