# Build the 3 Models
# ============================================================
#   - Load MobileNetV2 with ImageNet pretrained weights
#   - Replace ONLY the final classifier layer
#   - One single Linear layer — nothing else added
#   - No freezing — all parameters train from the start
#
# Same pattern applied to ResNet50 and EfficientNetB0
# so the training algorithm is identical for all 3 models.
# ============================================================

import torch.nn as nn
import torchvision.models as models


def build_model(model_name, cfg):
    """
      - Pretrained ImageNet weights
      - Replace only the final layer with our class count
      - No freezing — all layers train
      - Simple single Linear output layer
    """

    num_classes = cfg["NUM_CLASSES"]

    # ── MobileNetV2 ────────────────────────────────────────
    # 38 for disease classes
    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        # Replace final layer
        model.classifier[1] = nn.Linear(
            model.last_channel,   # 1280
            num_classes           # 38
        )

    # ── ResNet50 ───────────────────────────────────────────
    # Same idea — replace only the final fc layer
    # fc was Linear(2048 - 1000), now Linear(2048 - 38)
    elif model_name == "ResNet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Linear(
            model.fc.in_features,   # 2048
            num_classes             # 38
        )

    # ── EfficientNetB0 ─────────────────────────────────────
    # classifier is Sequential(Dropout, Linear(1280 - 1000))
    # Replace Linear part only
    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,   # 1280
            num_classes                         # 38
        )

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose: ResNet50, EfficientNetB0, MobileNetV2"
        )

    model = model.to(cfg["DEVICE"])

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {trainable:,}")

    return {"model": model, "name": model_name}