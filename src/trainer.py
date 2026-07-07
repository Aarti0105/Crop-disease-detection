# =============================================================================
# trainer.py  —  Model Training Pipeline 
# =============================================================================
#
# PURPOSE:
#   Handles everything that happens DURING training.
#   Works identically for all 3 models (ResNet50, EfficientNetB0, MobileNetV2).
#
#
#   PyTorch: You write the training loop yourself:
#            for epoch in range(epochs):
#                for batch in dataloader:
#                    predictions = model(batch)
#                    loss = criterion(predictions, labels)
#                    loss.backward()
#                    optimizer.step()
#            This is more code but gives you full control and understanding.
#
# PYTORCH TRAINING LOOP — 4 KEY STEPS PER BATCH:
#   1. optimizer.zero_grad()  → clear gradients from last batch
#   2. outputs = model(images)→ forward pass (make predictions)
#   3. loss.backward()        → backprop (compute how to adjust weights)
#   4. optimizer.step()       → update weights
#
# TRAIN vs EVAL MODE:
#   model.train() → enables Dropout and BatchNorm in training mode
#   model.eval()  → disables Dropout, uses running stats for BatchNorm
#   You MUST switch between these — forgetting model.eval() gives wrong results.
#
# WHAT REPLACES KERAS CALLBACKS:
#   EarlyStopping  → our EarlyStopping class below
#   ModelCheckpoint→ torch.save() when val_accuracy improves
#   ReduceLROnPlateau → torch.optim.lr_scheduler.ReduceLROnPlateau
# =============================================================================

import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model_builder import unfreeze_for_finetuning


# =============================================================================
# CLASS: EarlyStopping
# =============================================================================
class EarlyStopping:
    """
    Stops training when val_accuracy does not improve for PATIENCE epochs.
    Also saves the best model weights so we can restore them after stopping.

    CONCEPT:
        Without early stopping, a model might overfit if you train too long.
        Overfitting = high train accuracy, low val accuracy.
        Early stopping watches val_accuracy and stops when it plateaus.

    Usage:
        es = EarlyStopping(patience=5)
        for epoch in range(epochs):
            train(...)
            val_acc = validate(...)
            if es.step(val_acc, model):
                break
        es.restore_best(model)   # put back the best weights
    """

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0           # epochs since last improvement
        self.best_acc = 0.0         # best val_accuracy seen so far
        self.best_state = None        # deep copy of best model weights

    def step(self, val_acc, model):
        """
        Call once per epoch. Returns True if training should stop.

        Parameters
        ----------
        val_acc : float   validation accuracy this epoch (0 to 1)
        model   : nn.Module

        Returns
        -------
        bool : True = stop training,  False = continue
        """
        if val_acc > self.best_acc:
            # Improvement found — reset counter, save best weights
            self.best_acc = val_acc
            self.counter = 0
            # deep copy saves a snapshot of the weights right now
            # without this, self.best_state would update as the model changes
            self.best_state = copy.deepcopy(model.state_dict())
            return False   # do not stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True   # stop — no improvement for PATIENCE epochs
            return False

    def restore_best(self, model):
        """Reload the best weights into the model after training stops."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
            print(f"  Best weights restored (val_accuracy = {self.best_acc:.4f})")


# =============================================================================
# FUNCTION: train_one_epoch
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full pass through the training DataLoader.

    CONCEPT — What happens inside:
        For each batch of 32 images:
          1. Move images and labels to GPU/CPU
          2. Forward pass: model predicts class probabilities
          3. Compute loss: how wrong were the predictions?
          4. Backward pass: compute gradient (direction to adjust weights)
          5. Optimizer step: adjust weights slightly in the right direction

    Parameters
    ----------
    model     : nn.Module   in train() mode
    loader    : DataLoader  training DataLoader
    criterion : nn.Module   CrossEntropyLoss
    optimizer : Optimizer   Adam
    device    : str         "cuda" or "cpu"

    Returns
    -------
    avg_loss : float   average loss across all batches
    accuracy : float   fraction of correct predictions (0 to 1)
    """

    # Switch to train mode — enables Dropout and BatchNorm training behaviour
    model.train()

    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:

        # ── Move data to the same device as the model ──────────────────────
        # In PyTorch you must manually move tensors to GPU/CPU
        # If model is on GPU but data is on CPU → error
        images = images.to(device)
        labels = labels.to(device)

        # ── Step 1: Clear gradients from previous batch ────────────────────
        # Without this, gradients ACCUMULATE across batches → wrong updates
        optimizer.zero_grad()

        # ── Step 2: Forward pass ───────────────────────────────────────────
        # model(images) runs the images through all layers
        # outputs shape: (batch_size, num_classes) — raw scores (logits)
        outputs = model(images)

        # ── Step 3: Compute loss ───────────────────────────────────────────
        # CrossEntropyLoss compares logits to true labels
        # It applies softmax internally, so DO NOT add softmax to the model
        loss = criterion(outputs, labels)

        # ── Step 4: Backward pass ──────────────────────────────────────────
        # Computes how much each weight contributed to the error
        # Stores the gradient in each parameter's .grad attribute
        loss.backward()

        # ── Step 5: Update weights ─────────────────────────────────────────
        # Adam uses the gradients to nudge each weight in the right direction
        optimizer.step()

        # ── Track statistics ───────────────────────────────────────────────
        running_loss += loss.item()

        # argmax across class dimension gives predicted class index
        predicted = outputs.argmax(dim=1)
        correct  += predicted.eq(labels).sum().item()
        total    += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct / total

    return avg_loss, accuracy


# =============================================================================
# FUNCTION: validate
# =============================================================================
def validate(model, loader, criterion, device):
    """
    Run one pass through the validation DataLoader — no weight updates.

    KEY DIFFERENCES FROM train_one_epoch:
        model.eval()    → disables Dropout (all neurons active for inference)
        torch.no_grad() → tells PyTorch not to compute gradients (saves memory)
        NO optimizer.zero_grad() / loss.backward() / optimizer.step()

    Parameters
    ----------
    (same as train_one_epoch)

    Returns
    -------
    avg_loss : float
    accuracy : float
    """

    # Switch to evaluation mode — disables Dropout and uses BatchNorm running stats
    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    # torch.no_grad() tells PyTorch: do NOT track gradients for this block
    # This saves memory and makes inference faster
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs  = model(images)
            loss     = criterion(outputs, labels)

            running_loss += loss.item()
            predicted     = outputs.argmax(dim=1)
            correct      += predicted.eq(labels).sum().item()
            total        += labels.size(0)

    return running_loss / len(loader), correct / total


# =============================================================================
# FUNCTION: run_training_phase
# =============================================================================
def run_training_phase(model, loaders, optimizer, criterion, scheduler,
                       num_epochs, patience, device, save_path, phase_name):
    """
    Run one training phase (Phase 1 or Phase 2) with early stopping.

    Returns the history dict and the model with best weights restored.
    """

    early_stopping = EarlyStopping(patience=patience)

    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}

    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )

        # ── Validate ───────────────────────────────────────────────────────
        val_loss, val_acc = validate(
            model, loaders["val"], criterion, device
        )

        # ── LR Scheduler ───────────────────────────────────────────────────
        # ReduceLROnPlateau halves LR if val_loss does not improve for 3 epochs
        scheduler.step(val_loss)

        # ── Record history ─────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ── Save checkpoint if best so far ─────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        # ── Print epoch summary ─────────────────────────────────────────────
        print(f"  {phase_name}  Epoch {epoch:3d}/{num_epochs}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # ── Early stopping check ────────────────────────────────────────────
        if early_stopping.step(val_acc, model):
            print(f"  Early stopping at epoch {epoch}  (no improvement for {patience} epochs)")
            break

    # Restore the best weights seen during this phase
    early_stopping.restore_best(model)

    return model, history


# =============================================================================
# FUNCTION: save_history
# =============================================================================
def save_history(history, model_name, dataset_label, results_dir):
    """Save training history to JSON for plotting later."""

    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{model_name}_{dataset_label}_history.json")

    serialisable = {k: [float(v) for v in vals] for k, vals in history.items()}

    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"  History saved → {path}")


# =============================================================================
# MAIN FUNCTION: train_model
# =============================================================================
def train_model(model_bundle, loaders, dataset_label, config):
    """
    Full two-phase training pipeline for any of the 3 models.

    Phase 1 — Train head only (base frozen)
    Phase 2 — Fine-tune last N layers (base partially unfrozen)

    Parameters
    ----------
    model_bundle  : dict from build_model() — keys: "model", "name"
    loaders       : dict from prepare_data() — keys: "train", "val", "test"
    dataset_label : str  "baseline", "distorted", or "augmented"
    config        : dict from config.py

    Returns
    -------
    model   : trained PyTorch model with best weights loaded
    history : dict  combined Phase 1 + Phase 2 training history
    """

    model      = model_bundle["model"]
    model_name = model_bundle["name"]
    device     = config["DEVICE"]

    print()
    print("=" * 55)
    print(f"  TRAINING : {model_name}  |  {dataset_label}")
    print(f"  Device   : {device}")
    print("=" * 55)

    # Move model to GPU/CPU
    # This MUST happen before creating the optimizer
    model = model.to(device)

    # Loss function — CrossEntropyLoss for multi-class classification
    # Internally applies LogSoftmax then NLLLoss
    # Expects raw logits (no softmax) from the model — that is why our head
    # has no activation on its final layer
    criterion = nn.CrossEntropyLoss()

    # Checkpoint path — saved whenever val_accuracy improves
    os.makedirs(config["MODELS_DIR"], exist_ok=True)
    checkpoint_path = os.path.join(
        config["MODELS_DIR"],
        f"{model_name}_{dataset_label}_best.pth"   # .pth is the standard PyTorch extension
    )

    # =========================================================================
    # PHASE 1 — Train head only (base fully frozen)
    # =========================================================================
    print()
    print(f"  Phase 1 — Training head only  ({config['PHASE1_EPOCHS']} epochs max)")
    print(f"  Base is FROZEN — only classification head trains")
    print()

    # Optimizer only updates parameters where requires_grad=True
    # In Phase 1, only the head parameters have requires_grad=True
    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config["LEARNING_RATE"]
    )

    # ReduceLROnPlateau: halve LR when val_loss stops improving for 3 epochs
    scheduler_p1 = ReduceLROnPlateau(
        optimizer_p1, mode="min", factor=0.5, patience=3, verbose=True
    )

    model, history_p1 = run_training_phase(
        model       = model,
        loaders     = loaders,
        optimizer   = optimizer_p1,
        criterion   = criterion,
        scheduler   = scheduler_p1,
        num_epochs  = config["PHASE1_EPOCHS"],
        patience    = config["PATIENCE"],
        device      = device,
        save_path   = checkpoint_path,
        phase_name  = "Ph1",
    )

    best_p1 = max(history_p1["val_acc"])
    print(f"\n  Phase 1 complete — best val_accuracy: {best_p1:.4f}")

    # =========================================================================
    # PHASE 2 — Fine-tune last N layers at a smaller learning rate
    # =========================================================================
    print()
    print(f"  Phase 2 — Fine-tuning last {config['FINETUNE_LAYERS']} parameter groups")
    print(f"  Learning rate: {config['FINETUNE_LR']}  (smaller to protect pretrained weights)")
    print()

    # Unfreeze last N layers — sets their requires_grad=True
    model = unfreeze_for_finetuning(model, config)

    # Rebuild optimizer to include the newly unfrozen parameters
    # If we kept the old optimizer it would not know about the new parameters
    optimizer_p2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = config["FINETUNE_LR"]
    )

    scheduler_p2 = ReduceLROnPlateau(
        optimizer_p2, mode="min", factor=0.5, patience=3, verbose=True
    )

    model, history_p2 = run_training_phase(
        model       = model,
        loaders     = loaders,
        optimizer   = optimizer_p2,
        criterion   = criterion,
        scheduler   = scheduler_p2,
        num_epochs  = config["PHASE2_EPOCHS"],
        patience    = config["PATIENCE"],
        device      = device,
        save_path   = checkpoint_path,
        phase_name  = "Ph2",
    )

    best_p2 = max(history_p2["val_acc"])
    print(f"\n  Phase 2 complete — best val_accuracy: {best_p2:.4f}")

    # ── Merge Phase 1 + Phase 2 histories ─────────────────────────────────
    merged_history = {
        key: history_p1[key] + history_p2[key]
        for key in history_p1
    }

    # ── Save history ───────────────────────────────────────────────────────
    save_history(merged_history, model_name, dataset_label, config["RESULTS_DIR"])

    print()
    print(f"  Training complete — model saved to: {checkpoint_path}")
    print()

    return model, merged_history