# All Project Settings  #Stores every project configuration in one place.
# ============================================================

import os
import torch

BASE_PATH = "/Users/manish/Documents/Semister 2/7. Project/Aarti Data Droplet"


# Auto detect device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

CFG = {

    # Paths
    "CLEAN_IMAGES" : os.path.join(BASE_PATH, "images.npy"),
    "CLEAN_LABELS" : os.path.join(BASE_PATH, "labels.npy"),
    "DROP_IMAGES" : os.path.join(BASE_PATH, "droplet_images.npy"),
    "DROP_LABELS" : os.path.join(BASE_PATH, "droplet_labels.npy"),
    "MODELS_DIR" : os.path.join(BASE_PATH, "models"),
    "RESULTS_DIR" : os.path.join(BASE_PATH, "results"),

    # Device
    "DEVICE" : DEVICE,

    # Dataset
    "NUM_CLASSES" : 38,
    "BATCH_SIZE" : 64,
    "NUM_WORKERS" : 0,
    "TEST_SIZE" : 0.20,
    "VAL_SIZE" : 0.20,
    "SEED" : 42,
    "IMG_MEAN" : [0.485, 0.456, 0.406],
    "IMG_STD" : [0.229, 0.224, 0.225],

    # Models
    "MODELS" : ["MobileNetV2", "EfficientNetB0", "ResNet50"],

    # Training
    #   - Adam optimizer with lr=0.0001
    #   - Standard CrossEntropyLoss
    #   - No phases, simple loop
    "LR"           : 0.0001,
    "WEIGHT_DECAY" : 0.0001,
    "EPOCHS"       : 20,
    "PATIENCE"     : 5,
    "SETTINGS"     : ["baseline", "distorted", "augmented"],
    "MAX_CLASS_WEIGHT" : 3.0,
}