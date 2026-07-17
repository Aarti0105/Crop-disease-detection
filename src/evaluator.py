# Evaluate and Compare Models
# ============================================================
# What this file provides:
#   evaluate_model  → test accuracy, per-class F1, confusion matrix
#   plot_history    → training curve (accuracy and loss vs epoch)
#   load_model      → reload a saved .pth model from disk
#   run_cross_test  → test every model on both clean and droplet images
#   plot_final_chart → grouped bar chart for all results
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ──────────────────────────────────────────────────────────────
# get_predictions  —  run model and collect results
# ──────────────────────────────────────────────────────────────
def get_predictions(model, loader, device):
    """
    Pass all test images through the model and return predictions.

    torch.no_grad() tells PyTorch not to track gradients.
    We only need the forward pass here — no training.
    This saves memory and runs faster.

    Returns predictions and true labels as numpy arrays
    (sklearn metrics only accept numpy, not PyTorch tensors).
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            preds   = outputs.argmax(dim=1)          # highest score = predicted class
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ──────────────────────────────────────────────────────────────
# evaluate_model  —  full evaluation on test set
# ──────────────────────────────────────────────────────────────
def evaluate_model(model, loaders, class_names, model_name, setting, cfg):
    """
    Evaluate the model on the test set.
    Prints accuracy + per-class metrics.
    Shows and saves confusion matrix.

    Precision : of all images predicted as class X, how many were actually X?
    Recall    : of all images that ARE class X, how many did we find?
    F1-score  : combines Precision and Recall into one number (higher = better)
    """
    device = cfg["DEVICE"]
    model  = model.to(device)

    print(f"\n{'='*55}")
    print(f"  Evaluation: {model_name}  |  {setting}")
    print(f"{'='*55}")

    y_pred, y_true = get_predictions(model, loaders["test"], device)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n  Test Accuracy: {acc * 100:.2f}%\n")

    # Per-class breakdown
    report = classification_report(
        y_true, y_pred,
        target_names = class_names,
        digits       = 4,
    )
    print(report)

    # Confusion matrix
    _plot_confusion_matrix(y_true, y_pred, class_names, model_name, setting, cfg)

    # Save results to disk
    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)

    json_path = os.path.join(cfg["RESULTS_DIR"], f"{model_name}_{setting}_result.json")
    with open(json_path, "w") as f:
        json.dump({"model": model_name, "setting": setting, "accuracy": float(acc)}, f)

    txt_path = os.path.join(cfg["RESULTS_DIR"], f"{model_name}_{setting}_report.txt")
    with open(txt_path, "w") as f:
        f.write(f"Model: {model_name}  |  Setting: {setting}  |  Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)

    return float(acc)


# ──────────────────────────────────────────────────────────────
# _plot_confusion_matrix  —  heatmap of predictions vs truth
# ──────────────────────────────────────────────────────────────
def _plot_confusion_matrix(y_true, y_pred, class_names, model_name, setting, cfg):
    """
    Normalised confusion matrix — each row sums to 1.0.
    Diagonal = correctly predicted. Off-diagonal = mistakes.

    plt.savefig() before plt.show() — saves to disk first,
    then displays inline. Never call plt.close() before plt.show()
    or the chart won't appear in the notebook.
    """
    # Shorten long class names for axis labels
    short_names = [
        n.replace("Tomato_",       "T-")
         .replace("Tomato__",      "T-")
         .replace("Potato___",     "P-")
         .replace("Pepper__bell___","PP-")
         .replace("_", " ")
        for n in class_names
    ]

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names,
        linewidths=0.4, ax=ax,
    )
    ax.set_title(f"{model_name}  |  {setting} — Confusion Matrix",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True",      fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(cfg["RESULTS_DIR"], f"cm_{model_name}_{setting}.png")
    plt.show()


# ──────────────────────────────────────────────────────────────
# plot_history  —  training curves
# ──────────────────────────────────────────────────────────────
def plot_history(model_name, setting, cfg):
    """
    Load saved training history and plot accuracy + loss curves.

    Good pattern: train and val lines rise together and stay close.
    Bad pattern:  train keeps rising but val stays flat = overfitting.
    """
    path = os.path.join(cfg["RESULTS_DIR"], f"{model_name}_{setting}_history.json")

    if not os.path.exists(path):
        print(f"  No history file at {path} — run training first")
        return

    h      = json.load(open(path))
    epochs = range(1, len(h["train_acc"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves  —  {model_name}  ({setting})",
                 fontsize=13, fontweight="bold")

    ax1.plot(epochs, h["train_acc"],  label="Train", color="#2196F3", linewidth=2)
    ax1.plot(epochs, h["val_acc"],    label="Val",   color="#F44336", linewidth=2, linestyle="--")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, h["train_loss"], label="Train", color="#2196F3", linewidth=2)
    ax2.plot(epochs, h["val_loss"],   label="Val",   color="#F44336", linewidth=2, linestyle="--")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(cfg["RESULTS_DIR"], f"curve_{model_name}_{setting}.png")
    plt.show()


# ──────────────────────────────────────────────────────────────
# load_model  —  reload a saved model from disk
# ──────────────────────────────────────────────────────────────
def load_model(model_name, setting, cfg):
    """
    Reload a trained model from a .pth file.

    We first rebuild the architecture with build_model(),
    then load the saved weights into it with load_state_dict().
    This is how PyTorch model loading works — the file only
    stores weights, not the model structure.
    """
    from model_builder import build_model

    path = os.path.join(cfg["MODELS_DIR"], f"{model_name}_{setting}_best.pth")

    if not os.path.exists(path):
        print(f"  Not found: {path}  (run training first)")
        return None

    bundle = build_model(model_name, cfg)
    model  = bundle["model"]
    model.load_state_dict(torch.load(path, map_location=cfg["DEVICE"]))
    model  = model.to(cfg["DEVICE"])
    model.eval()

    print(f"  Loaded: {model_name}_{setting}_best.pth")
    return model


# ──────────────────────────────────────────────────────────────
# run_cross_test  —  test every model on both datasets
# ──────────────────────────────────────────────────────────────
def run_cross_test(trained_models, loaders_clean, loaders_drop,
                   y_clean, y_drop, cfg):
    """
    The core dissertation experiment.

    Tests every model on BOTH clean and droplet test images,
    regardless of what it was trained on.

    Key metrics:
    Robustness Gap   = how much accuracy drops on wet images
    Augmentation Gap = how much combined training recovers that drop

    Returns a results dictionary with all accuracy values.
    """
    device  = cfg["DEVICE"]
    results = {}

    print(f"\n{'='*65}")
    print("  CROSS-DATASET ROBUSTNESS TEST")
    print(f"{'='*65}")
    print(f"\n  {'Model':<18}  {'Setting':<12}  {'On Clean':>9}  {'On Droplet':>11}")
    print("  " + "-"*56)

    for model_name, settings_dict in trained_models.items():
        results[model_name] = {}

        for setting_name, model in settings_dict.items():
            if model is None:
                continue

            model = model.to(device)

            # Test on clean images
            y_pred_clean, _ = get_predictions(model, loaders_clean["test"], device)
            acc_clean       = accuracy_score(y_clean, y_pred_clean)

            # Test on droplet images
            y_pred_drop, _  = get_predictions(model, loaders_drop["test"], device)
            acc_drop        = accuracy_score(y_drop, y_pred_drop)

            results[model_name][setting_name] = {
                "clean"   : float(acc_clean),
                "droplet" : float(acc_drop),
            }

            print(f"  {model_name:<18}  {setting_name:<12}  "
                  f"{acc_clean*100:>8.2f}%  {acc_drop*100:>10.2f}%")
        print()

    # Compute and display gap metrics
    print(f"\n  {'Model':<18}  {'Robustness Gap':>15}  {'Augmentation Gap':>17}")
    print("  " + "-"*55)

    for model_name in results:
        r = results[model_name]
        if "baseline" not in r or "augmented" not in r:
            continue

        robustness_gap   = r["baseline"]["clean"]    - r["baseline"]["droplet"]
        augmentation_gap = r["augmented"]["droplet"] - r["baseline"]["droplet"]

        results[model_name]["robustness_gap"]   = robustness_gap
        results[model_name]["augmentation_gap"] = augmentation_gap

        print(f"  {model_name:<18}  {robustness_gap*100:>+14.2f}%  {augmentation_gap*100:>+16.2f}%")

    print("\n  Robustness Gap  : accuracy drop on wet leaves (smaller = more robust)")
    print("  Augmentation Gap: drop recovered by augmented training (larger = better)")

    # Save all results
    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)
    out_path = os.path.join(cfg["RESULTS_DIR"], "cross_test_results.json")
    with open(out_path, "w") as f:
        serialisable = {}
        for m, d in results.items():
            serialisable[m] = {k: {kk: float(vv) for kk, vv in v.items()}
                               if isinstance(v, dict) else float(v)
                               for k, v in d.items()}
        json.dump(serialisable, f, indent=2)

    return results


# ──────────────────────────────────────────────────────────────
# plot_final_chart  —  grouped bar chart for all results
# ──────────────────────────────────────────────────────────────
def plot_final_chart(cfg):
    """
    Grouped bar chart showing all 3 models × all 3 settings.
    Loads accuracy values from the saved result JSON files.
    """
    model_list   = ["ResNet50", "EfficientNetB0", "MobileNetV2"]
    setting_list = ["baseline", "distorted", "augmented"]
    colours      = {"baseline": "#2196F3", "distorted": "#F44336", "augmented": "#4CAF50"}

    data = {}
    for m in model_list:
        data[m] = {}
        for s in setting_list:
            p = os.path.join(cfg["RESULTS_DIR"], f"{m}_{s}_result.json")
            data[m][s] = json.load(open(p))["accuracy"] * 100 if os.path.exists(p) else 0

    x     = np.arange(len(model_list))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, setting in enumerate(setting_list):
        values = [data[m][setting] for m in model_list]
        bars   = ax.bar(x + i * width, values, width,
                        label=setting.capitalize(),
                        color=colours[setting],
                        edgecolor="black", linewidth=0.7)

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{v:.1f}%",
                    ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_list, fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "Final Results — All Models × All Settings",
        fontsize=13, fontweight="bold",
    )
    ax.axhline(y=95, color="green", linestyle="--", linewidth=1.5,
               alpha=0.7, label="95% target")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(cfg["RESULTS_DIR"], "final_chart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()