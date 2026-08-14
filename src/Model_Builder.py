

# Model Builder

#---------------------------------------------------------------------------------------------------------------------

#Section 1 - Import Libraries

import torch.nn as nn
import torchvision.models as models

#---------------------------------------------------------------------------------------------------------------------

# Section 2 - Build Model Function

def build_model(model_name, cfg):
    num_classes = cfg["NUM_CLASSES"]

    # MobileNetV2
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

    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )
        # Replace final layer
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )

    elif model_name == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(
            weights=models.ShuffleNet_V2_X1_0_Weights.DEFAULT
        )
        # Replace final layer
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )   
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported models: MobileNetV2, EfficientNetB0, ShuffleNetV2"
        )
    model = model.to(cfg["DEVICE"])

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters : {trainable:,}")

    return {"model": model, "name": model_name}

#---------------------------------------------------------------------------------------------------------------------