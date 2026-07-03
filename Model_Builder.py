# =============================================================================
# model_builder.py  —  Model Architecture Definitions  (PyTorch version)
# =============================================================================
#
# PURPOSE:
#   Defines ResNet50, EfficientNetB0 and MobileNetV2 using PyTorch/torchvision.
#   Same three models as the literature requires, same Transfer Learning strategy.

#
#   PyTorch: replace the final classification layer(s) of the pretrained model
#            with our own custom head using nn.Sequential.
#            The optimizer and loss function live OUTSIDE the model object.
#            We create them in trainer.py.
#
# FREEZING PARAMETERS IN PYTORCH:
#   Every parameter (weight tensor) has a flag: requires_grad
#   requires_grad=True  → this weight will be updated during training
#   requires_grad=False → this weight is FROZEN, gradient is not computed
#
#   Phase 1: set requires_grad=False for all base params, True for head only
#   Phase 2: unfreeze last N params of the base for fine-tuning
#
# WHERE THE CLASSIFIER HEAD LIVES:
#   ResNet50     →  model.fc         (the final fully connected layer)
#   EfficientNetB0 →  model.classifier (the final block)
#   MobileNetV2  →  model.classifier (the final block)
# =============================================================================

import torch
import torch.nn as nn
import torchvision.models as models


# =============================================================================
# SHARED HELPER — Custom classification head
# =============================================================================
def _build_head(in_features, num_classes):
    """
    Build a custom classification head — identical for all 3 models.

    Replaces the original 1000-class ImageNet head with our 16-class head.

    Architecture:
        Linear(in_features → 256)  ReLU  Dropout(0.4)
        Linear(256 → 128)          ReLU  Dropout(0.3)
        Linear(128 → num_classes)         ← no activation here!

    WHY NO SOFTMAX AT THE END?
        PyTorch's nn.CrossEntropyLoss combines LogSoftmax + NLLLoss internally.
        Adding a Softmax before it would cause numerical errors.
        We only apply softmax AFTER training when we need probabilities
        (for evaluation — done with torch.softmax() in evaluator.py).

    Parameters
    ----------
    in_features : int   size of the feature vector from the base model
    num_classes : int   number of disease classes (16)
    """
    return nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
        # NO Softmax here — CrossEntropyLoss handles it internally
    )


# =============================================================================
# MODEL 1 — ResNet50
# =============================================================================
def build_resnet50(config):
    """
    Load pretrained ResNet50 and replace its final layer with our head.

    ResNet50 structure:
        conv1 → bn1 → relu → maxpool
        layer1 → layer2 → layer3 → layer4   (residual blocks)
        avgpool                              (global average pooling)
        fc                                   ← WE REPLACE THIS

    The original fc is: Linear(2048, 1000)
    We replace it with: our custom head starting from 2048 features.

    Freezing Phase 1:
        All params in conv1/bn1/layer1/.../layer4  → requires_grad=False
        All params in fc (our new head)             → requires_grad=True
    """
    print("  Loading ResNet50 (ImageNet pretrained)...")

    # IMAGENET1K_V2 = best available ImageNet weights for ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Get the number of input features to the original fc layer
    in_features = model.fc.in_features   # = 2048 for ResNet50

    # Replace the original 1000-class fc with our 16-class head
    model.fc = _build_head(in_features, config["NUM_CLASSES"])

    # PHASE 1 — Freeze everything EXCEPT the new fc head
    # Freeze all base parameters
    for name, param in model.named_parameters():
        if "fc" not in name:          # "fc" = our custom head
            param.requires_grad = False
        else:
            param.requires_grad = True  # head stays trainable

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Total params     : {total:,}")
    print(f"  Trainable (Ph.1) : {trainable:,}  (head only — base frozen)")
    print()

    return model


# =============================================================================
# MODEL 2 — EfficientNetB0
# =============================================================================
def build_efficientnetb0(config):
    """
    Load pretrained EfficientNetB0 and replace its classifier.

    EfficientNetB0 structure:
        features   (blocks 0-8)   ← feature extractor
        avgpool                   ← global average pooling
        classifier                ← WE REPLACE THIS
            Sequential(Dropout(0.2), Linear(1280, 1000))

    The original classifier outputs 1000 classes from 1280 features.
    We replace it with our custom 16-class head.
    """
    print("  Loading EfficientNetB0 (ImageNet pretrained)...")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Get input features of the original Linear layer inside classifier
    # model.classifier is Sequential(Dropout, Linear(1280, 1000))
    in_features = model.classifier[1].in_features   # = 1280

    # Replace the entire classifier block
    model.classifier = _build_head(in_features, config["NUM_CLASSES"])

    # PHASE 1 — Freeze everything except classifier
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Total params     : {total:,}")
    print(f"  Trainable (Ph.1) : {trainable:,}  (head only — base frozen)")
    print()

    return model


# =============================================================================
# MODEL 3 — MobileNetV2
# =============================================================================
def build_mobilenetv2(config):
    """
    Load pretrained MobileNetV2 and replace its classifier.

    MobileNetV2 structure:
        features   (18 inverted residual blocks)  ← feature extractor
        classifier                                 ← WE REPLACE THIS
            Sequential(Dropout(0.2), Linear(1280, 1000))

    Same structure as EfficientNetB0's classifier — 1280 input features.
    """
    print("  Loading MobileNetV2 (ImageNet pretrained)...")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)

    in_features = model.classifier[1].in_features   # = 1280

    model.classifier = _build_head(in_features, config["NUM_CLASSES"])

    # PHASE 1 — Freeze everything except classifier
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Total params     : {total:,}")
    print(f"  Trainable (Ph.1) : {trainable:,}  (head only — base frozen)")
    print()

    return model


# =============================================================================
# FUNCTION: unfreeze_for_finetuning
# =============================================================================
def unfreeze_for_finetuning(model, config):
    """
    Switch any model from Phase 1 (frozen base) to Phase 2 (fine-tuning).

    Works identically for all 3 model architectures — no model-specific code.

    What this does:
        1. Gets ALL named parameters as an ordered list
        2. Unfreezes the LAST FINETUNE_LAYERS of them
        3. The newly trainable params will be included in the optimizer
           (the optimizer is rebuilt in trainer.py with the updated param list)

    WHY UNFREEZE FROM THE END?
        In all three architectures, the LAST layers are the most task-specific.
        They learn high-level patterns (disease spots, textures).
        The FIRST layers learn general features (edges, colours) — we keep those frozen.
        Unfreezing from the end gives the most benefit for fine-tuning.

    Parameters
    ----------
    model  : PyTorch model (after Phase 1 training)
    config : dict from config.py

    Returns
    -------
    model : same model with last N params now set to requires_grad=True
    """
    n = config["FINETUNE_LAYERS"]

    # Get all (name, parameter) pairs as a flat list
    all_params = list(model.named_parameters())

    # Unfreeze the LAST n parameter tensors
    for name, param in all_params[-n:]:
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Phase 2 — Unfroze last {n} parameter tensors")
    print(f"  Trainable (Ph.2) : {trainable:,}")
    print()

    return model


# =============================================================================
# MASTER FUNCTION: build_model
# =============================================================================
def build_model(model_name, config):
    """
    Single entry point — build any model by name.

    All training scripts call this. They never import individual
    build functions. This keeps the training loop identical for all models.

    Parameters
    ----------
    model_name : str   "ResNet50", "EfficientNetB0", or "MobileNetV2"
    config     : dict  from config.py

    Returns
    -------
    dict:
        "model"  → PyTorch nn.Module  (Phase 1 ready — base frozen)
        "name"   → model name string
    """
    print()
    print("=" * 55)
    print(f"  MODEL BUILDER — {model_name}")
    print("=" * 55)

    builders = {
        "ResNet50"       : build_resnet50,
        "EfficientNetB0" : build_efficientnetb0,
        "MobileNetV2"    : build_mobilenetv2,
    }

    if model_name not in builders:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(builders.keys())}"
        )

    model = builders[model_name](config)

    return {"model": model, "name": model_name}