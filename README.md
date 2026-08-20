# Crop Disease Detection Under Realistic Environmental Distortions Using Synthetic Water Droplet Augmentation

**Author:** Aarti Bandgar  
**Institution:** University of Galway, Ireland  
**Programme:** MSc Computer Science - Data Analytics  

---

## What This Project Is About

Deep learning models for crop disease detection achieve very high accuracy on clean laboratory images but fail badly when tested on wet leaf images in real field conditions. This project investigates that problem.

A **synthetic water droplet augmentation pipeline** is built that adds realistic water droplets to clean PlantVillage leaf images. Three lightweight CNN models — **MobileNetV2**, **EfficientNetB0**, and **ShuffleNetV2** — are trained under two settings:

- **Baseline** — trained and tested on clean images
- **Distorted** — trained and tested on synthetic droplet images

A **cross-dataset robustness test** is then run where each clean-trained (Baseline) model is evaluated on wet leaf images to measure how much accuracy drops. Results show drops of **31 to 41 percentage points**, confirming that water droplets are a serious threat to real-world disease detection.

---

## Key Results

| Model | Baseline Accuracy | Distorted Accuracy | Drop on Wet Leaves |
|---|---|---|---|
| MobileNetV2 | 99.59% | 99.37% | −36.06% |
| EfficientNetB0 | 99.77% | 99.55% | −40.75% |
| ShuffleNetV2 | 99.61% | 99.12% | −31.67% |

> The drop column shows what happens when a clean-trained model is tested on wet leaf images it has never seen; this is the core finding of this project.

---

## Dataset

**PlantVillage Dataset** — 54,305 colour images across 38 disease classes  
Downloaded from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

## Project Structure

```
Crop-disease-detection/
│
├── src/                          # All source code (pipeline modules)
│   ├── config.py                 # All project settings in one place (paths, hyperparameters, model names)
│   ├── data_loader.py            # Loads images, encodes labels, splits into train/val/test, returns DataLoaders
│   ├── model_builder.py          # Builds MobileNetV2, EfficientNetB0, ShuffleNetV2 with pretrained ImageNet weights
│   ├── trainer.py                # Training loop with Adam optimizer, early stopping, weighted loss, ReduceLROnPlateau
│   ├── evaluator.py              # Evaluation functions — accuracy, precision, recall, F1, confusion matrix, Grad-CAM
│
├── notebooks/                    # Jupyter notebooks — run these in order
│   ├── 01_Dataset.ipynb          # Load PlantVillage images from folder and save as images.npy and labels.npy
│   ├── 02_Synthetic_Droplet_Generation.ipynb   # Apply water droplet augmentation to clean images, save as droplet_images.npy
│   ├── 03_Training.ipynb         # Train all 3 models under Baseline and Distorted settings (6 training runs total)
│   └── 04_Evaluation.ipynb       # Evaluate all models — accuracy, metrics, confusion matrix, cross-dataset test, Grad-CAM
│
└── README.md                     # This file
```

---

## How to Run

### Step 1 — Install requirements

```bash
pip install torch torchvision scikit-learn numpy opencv-python matplotlib seaborn pytorch-grad-cam
```

### Step 2 — Download the dataset

Download the PlantVillage dataset (colour version) from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) and unzip it.

### Step 3 — Set your paths

Open `src/config.py` and update `BASE_PATH` to your local project folder:

```python
BASE_PATH = "/your/path/to/project"
```

### Step 4 — Run notebooks in order

| Notebook | What it does |
|---|---|
| `01_Dataset.ipynb` | Loads all PlantVillage images and saves them as `.npy` files |
| `02_Synthetic_Droplet_Generation.ipynb` | Generates synthetic water droplet images |
| `03_Training.ipynb` | Trains all 3 models under both settings |
| `04_Evaluation.ipynb` | Evaluates all models and produces all results |

---

## What Each Source File Does

| File | Purpose |
|---|---|
| `config.py` | Central settings file — change paths, hyperparameters, and model names here |
| `data_loader.py` | Handles all data loading, label encoding, and train/val/test splitting |
| `model_builder.py` | Loads pretrained models and replaces the final layer for 38-class classification |
| `trainer.py` | Runs the training loop — weighted loss, Adam, early stopping, learning rate scheduling |
| `evaluator.py` | Computes all evaluation metrics and produces confusion matrix and comparison tables |

---

## Hardware Used

- Apple Silicon Mac (MPS GPU backend)
- PyTorch 2.x, torchvision, scikit-learn, OpenCV, matplotlib

---
