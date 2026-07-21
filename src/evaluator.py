

# Evaluate the Trained Model

#---------------------------------------------------------------------------------------------------------------------

# Section 1 - Import Libraries

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from model_builder import build_model

#----------------------------------------------------------------------------------------------------------------------

# Section 2 - plot_history (learning curves)

def plot_history(model_name, setting, cfg):
    path = os.path.join(cfg["RESULTS_DIR"], f"{model_name}_{setting}_history.json")
    if not os.path.exists(path):
        print(f"No history found.")
        return
    h = json.load(open(path))
    epochs = range(1, len(h["train_acc"]) + 1)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, h["train_acc"], label="Train")
    plt.plot(epochs, h["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, h["train_loss"], label="Train")
    plt.plot(epochs, h["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

#----------------------------------------------------------------------------------------------------------------------

# Section 3 - get_predictions (run model on a loader)

def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds   = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

#----------------------------------------------------------------------------------------------------------------------

# Section 4 - Predict (run model on test loader and return predictions)

def predict(model, loader, cfg):
    device = cfg["DEVICE"]
    model  = model.to(device)
    y_pred, y_true = get_predictions(model, loader, device)
    print(f"Predictions ready - {len(y_true):,} test images")
    return y_pred, y_true

#------------------------------------------------------------------------------------------------------------------------

# Section 5 - Show test accuracy

def show_accuracy(y_pred, y_true, model_name, setting):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f" {model_name}  |  {setting.upper()}")
    print(f"{'='*50}")
    print(f"\n  Test Accuracy : {acc*100:.2f}%\n")

#-------------------------------------------------------------------------------------------------------------------------

# Section 6 - Show metrics

def show_metrics(y_pred, y_true, class_names, model_name, setting, cfg):
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    wt_p = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    wt_r = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    wt_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n Overall Metrics")
    print(f"{'─'*50}")
    print(f"{'Metric':<20}  {'Macro Avg':>10}  {'Weighted Avg':>13}")
    print(f"{'─'*50}")
    print(f"{'Precision':<20}  {macro_p*100:>9.2f}%  {wt_p*100:>12.2f}%")
    print(f"{'Recall':<20}  {macro_r*100:>9.2f}%  {wt_r*100:>12.2f}%")
    print(f"{'F1-Score':<20}  {macro_f1*100:>9.2f}%  {wt_f1*100:>12.2f}%")

    print(f"\n Per Class Metrics")
    print(f"{'─'*50}")
    report = classification_report(
        y_true, y_pred,
        target_names = class_names,
        digits = 4,
        zero_division= 0,
    )
    print(report)

#-------------------------------------------------------------------------------------------------------------------------

# Section 7 - Confusion metrics

def show_confusion(y_pred, y_true, class_names):
    cm = confusion_matrix(y_true, y_pred)
    total = cm.sum()

    tp = fp = fn = tn = 0
    for i in range(len(class_names)):
        tp += cm[i, i]
        fp += cm[:, i].sum() - cm[i, i]
        fn += cm[i, :].sum() - cm[i, i]
        tn += total - cm[i, i] - (cm[:, i].sum() - cm[i, i]) - (cm[i, :].sum() - cm[i, i])

    matrix = np.array([[tp, fp], [fn, tn]])

    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.axis("off")

    # Correct predictions
    correct = "#A5D6A7"
    # Incorrect predictions (red)
    incorrect = "#EF9A9A"

    boxes = [
        (0,1,"True Positive\n(TP)",tp,correct),
        (1,1,"False Positive\n(FP)",fp,incorrect),
        (0,0,"False Negative\n(FN)",fn,incorrect),
        (1,0,"True Negative\n(TN)",tn,correct),
    ]

    for x,y,label,value,color in boxes:
        rect = plt.Rectangle((x,y),1,1, acecolor=color, edgecolor="black", linewidth=2)
        ax.add_patch(rect)
        ax.text(x+0.5,y+0.60,label, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x+0.5,y+0.30,f"{value:,}", ha="center", va="center", fontsize=15)

    plt.title("Confusion Matrix",fontsize=16)
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------

# Section 8 - Load model

def load_model(model_name, setting, cfg):
    path = os.path.join(cfg["MODELS_DIR"], f"{model_name}_{setting}_best.pth")
    if not os.path.exists(path):
        print(f" Not found: {path}")
        return None
    bundle = build_model(model_name, cfg)
    model  = bundle["model"]
    model.load_state_dict(torch.load(path, map_location=cfg["DEVICE"]))
    model  = model.to(cfg["DEVICE"])
    model.eval()
    print(f" Loaded: {model_name}_{setting}_best.pth")
    return model

#-----------------------------------------------------------------------------------------------------------------------------