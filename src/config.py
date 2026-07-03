import os
import torch

# =============================================================================
# SECTION 1 — FILE PATHS
# =============================================================================


BASE_PATH = r"C:\Users\manish\Documents\Semister 2\7. Project\Aarti Data Droplet"

config = {

    # ── Input data paths ──────────────────────────────────────────────────
    # Clean dataset (original images without any droplets)
    "CLEAN_IMAGES_PATH" : os.path.join(BASE_PATH, "images.npy"),
    "CLEAN_LABELS_PATH" : os.path.join(BASE_PATH, "labels.npy"),

    # Droplet dataset (images with Version 1 synthetic water droplets)
    "DROPLET_IMAGES_PATH" : os.path.join(BASE_PATH, "droplet_images.npy"),
    "DROPLET_LABELS_PATH"  : os.path.join(BASE_PATH, "droplet_labels.npy"),

    # ── Output folders ────────────────────────────────────────────────────
    # Where trained models (.h5 files) are saved
    "MODELS_DIR"  : os.path.join(BASE_PATH, "models"),

    # Where training graphs and evaluation charts are saved
    "RESULTS_DIR" : os.path.join(BASE_PATH, "results"),


    "DEVICE" : "cuda" if torch.cuda.is_available() else "cpu",

    # ==========================================================================
    # SECTION 2 — IMAGE SETTINGS
    # ==========================================================================

    # Every image was resized to 224×224 pixels during preprocessing (Notebook 01)
    "IMAGE_SIZE"     : (224, 224),

    # 3 colour channels — Red, Green, Blue
    "IMAGE_CHANNELS" : 3,

    # Combined as a tuple: (height, width, channels)
    # This is the shape TensorFlow/Keras expects as input
    "INPUT_SHAPE"    : (3, 224, 224),  # PyTorch uses (C, H, W) format

    # Number of disease classes in PlantVillage (16 classes)
    "NUM_CLASSES"    : 16,

    # ImageNet normalisation values — required for all pretrained torchvision models
    # These are the mean and std of the ImageNet training set (per channel)
    # Every image MUST be normalised with these before passing to the model
    "IMAGENET_MEAN" : [0.485, 0.456, 0.406],
    "IMAGENET_STD"  : [0.229, 0.224, 0.225],
 


    # ==========================================================================
    # SECTION 3 — MODELS

    #   Phase 1 → freeze base, train top layers only
    #   Phase 2 → unfreeze last N layers, fine-tune with small LR
    #
    # This means the ALGORITHM IS IDENTICAL for all three models.
    # Only the base architecture (what is inside the frozen part) differs.
 
    "MODELS" : ["ResNet50", "EfficientNetB0", "MobileNetV2"],
 
 
    # ==========================================================================
    # SECTION 4 — EXPERIMENTAL SETTINGS
    # From Table II in proposed approach:
    #   Baseline  → clean only
    #   Distorted → droplet only
    #   Augmented → clean + droplet combined
    # ==========================================================================
 
    "SETTINGS" : ["baseline", "distorted", "augmented"],
 
 
    # ==========================================================================
    # SECTION 5 — TRAINING HYPERPARAMETERS
    # From proposed approach: "Adam/SGD optimizer, identical hyperparameters"
    # Adam is used here (standard, adaptive — better than SGD for transfer learning)
    # ==========================================================================
 
    "EPOCHS"       : 30,
    "BATCH_SIZE"   : 32,
    "LEARNING_RATE": 0.001,    # Phase 1 LR
    "FINETUNE_LR"  : 0.0001,   # Phase 2 LR — much smaller to protect pre-trained weights


    # DataLoader workers — number of CPU threads for loading batches in parallel
    # Set to 0 on Windows (avoids multiprocessing errors); 4 on Linux/Mac
    "NUM_WORKERS"   : 4,
 
 
    # ==========================================================================
    # SECTION 6 — TRAINING CONTROL
    # ==========================================================================
 
    # EarlyStopping: stop if val_accuracy does not improve for this many epochs
    "PATIENCE"      : 5,
 
    # Data split: 64% train / 16% val / 20% test
    # Same split used across all 3 settings for fair comparison
    "TEST_SIZE"     : 0.20,
    "VAL_SIZE"      : 0.20,
    "RANDOM_STATE"  : 42,
 
 
    # ==========================================================================
    # SECTION 7 — TRANSFER LEARNING SETTINGS
    # Phase 1: Freeze entire base → train only custom top layers
    # Phase 2: Unfreeze last FINETUNE_LAYERS → fine-tune at lower LR
    # ==========================================================================
 
    "PHASE1_EPOCHS"   : 15,
    "PHASE2_EPOCHS"   : 20,
    "FINETUNE_LAYERS" : 30,   # last 30 layers of base are unfrozen in Phase 2
 
}
 