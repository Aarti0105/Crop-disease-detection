# Train the Model
# ============================================================

import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight


class EarlyStopping:
    """Stop training when val accuracy stops improving. Save best weights."""

    def __init__(self, patience):
        self.patience   = patience
        self.counter    = 0
        self.best_acc   = 0.0
        self.best_state = None

    def check(self, val_acc, model):
        if val_acc > self.best_acc:
            self.best_acc   = val_acc
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)
            print(f"  Best weights restored  (val_acc = {self.best_acc:.4f})")


def get_weighted_loss(loaders, cfg, device):
    """
    Build CrossEntropyLoss with class weights.

    Minority classes get higher weight so the model cannot ignore them.
    Weight is capped at MAX_CLASS_WEIGHT (3.0) to keep training stable.

    WHY SAFE NOW AT LR=0.0001:
        Class weighting caused oscillation before at LR=0.001 because
        large weights created huge gradient spikes with a high LR.
        At LR=0.0001 each update step is 10x smaller, so the same
        weight difference causes 10x smaller spikes — completely stable.
    """
    train_labels = loaders["train"].dataset.labels
    weights      = compute_class_weight(
        class_weight = "balanced",
        classes = np.arange(cfg["NUM_CLASSES"]),
        y = train_labels,
    )

    # Cap extreme weights
    weights = np.clip(weights, 0, cfg["MAX_CLASS_WEIGHT"])
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    return nn.CrossEntropyLoss(weight=weight_tensor)


def train_model(bundle, loaders, setting_name, cfg):
    model      = bundle["model"]
    model_name = bundle["name"]
    device     = cfg["DEVICE"]

    # Weighted loss — fixes Potato___healthy and Tomato_mosaic_virus
    criterion = get_weighted_loss(loaders, cfg, device)
    print()

    # Adam optimizer — 
    optimizer = optim.Adam(
        model.parameters(),
        lr = cfg["LR"],           # 0.0001
        weight_decay = cfg["WEIGHT_DECAY"],
    )

    # Halve LR when val accuracy stops improving for 3 epochs
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=3, min_lr=1e-8,
    )

    os.makedirs(cfg["MODELS_DIR"], exist_ok=True)
    save_path = os.path.join(
        cfg["MODELS_DIR"],
        f"{model_name}_{setting_name}_best.pth"
    )

    early_stop = EarlyStopping(patience=cfg["PATIENCE"])
    history    = {"train_acc": [], "val_acc": [],
                  "train_loss": [], "val_loss": []}
    best_val   = 0.0

    for epoch in range(1, cfg["EPOCHS"] + 1):

        # Training phase 
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(loaders["train"])
        train_accuracy = 100 * correct / total

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0

        with torch.no_grad():
            for images, labels in loaders["val"]:
                images, labels  = images.to(device), labels.to(device)
                outputs         = model(images)
                val_loss_sum   += criterion(outputs, labels).item()
                _, predicted    = torch.max(outputs, 1)
                total          += labels.size(0)
                correct        += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss_sum / len(loaders["val"])

        scheduler.step(val_accuracy)

        # Record
        history["train_acc"].append(round(train_accuracy / 100, 4))
        history["val_acc"].append(round(val_accuracy / 100, 4))
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))

        if val_accuracy > best_val:
            best_val = val_accuracy
            torch.save(model.state_dict(), save_path)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:2d}/{cfg['EPOCHS']}]  "
            f"Train Loss: {avg_train_loss:.4f}  "
            f"Train Acc: {train_accuracy:.2f}%  "
            f"Val Acc: {val_accuracy:.2f}%  "
            f"lr={lr:.2e}"
        )
        print(f"  {'-'*75}")

        if early_stop.check(val_accuracy / 100, model):
            print(f"\n  Early stop — no improvement for {cfg['PATIENCE']} epochs")
            break

    early_stop.restore_best(model)

    os.makedirs(cfg["RESULTS_DIR"], exist_ok=True)
    history_path = os.path.join(
        cfg["RESULTS_DIR"], f"{model_name}_{setting_name}_history.json"
    )
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Best Validation Accuracy : {best_val:.2f}%")
    return model, history