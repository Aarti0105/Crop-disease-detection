import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import os

# ──────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Apple — Apple Scab",
    "Apple — Black Rot",
    "Apple — Cedar Apple Rust",
    "Apple — Healthy",
    "Blueberry — Healthy",
    "Cherry — Powdery Mildew",
    "Cherry — Healthy",
    "Corn — Cercospora Leaf Spot",
    "Corn — Common Rust",
    "Corn — Northern Leaf Blight",
    "Corn — Healthy",
    "Grape — Black Rot",
    "Grape — Esca (Black Measles)",
    "Grape — Leaf Blight",
    "Grape — Healthy",
    "Orange — Citrus Greening",
    "Peach — Bacterial Spot",
    "Peach — Healthy",
    "Pepper — Bacterial Spot",
    "Pepper — Healthy",
    "Potato — Early Blight",
    "Potato — Late Blight",
    "Potato — Healthy",
    "Raspberry — Healthy",
    "Soybean — Healthy",
    "Squash — Powdery Mildew",
    "Strawberry — Leaf Scorch",
    "Strawberry — Healthy",
    "Tomato — Bacterial Spot",
    "Tomato — Early Blight",
    "Tomato — Late Blight",
    "Tomato — Leaf Mold",
    "Tomato — Septoria Leaf Spot",
    "Tomato — Spider Mites",
    "Tomato — Target Spot",
    "Tomato — Yellow Leaf Curl Virus",
    "Tomato — Mosaic Virus",
    "Tomato — Healthy",
]

MODEL_CONFIG = {
    "MobileNetV2": {"params": "2.27M", "feature_dim": 1280},
    "EfficientNetB0": {"params": "4.06M", "feature_dim": 1280},
    "ShuffleNetV2": {"params": "1.29M", "feature_dim": 1024},
}

RESULTS = {
    "MobileNetV2":    {"baseline": 99.59, "distorted": 99.37, "wet_drop": 63.54, "gap": 36.06},
    "EfficientNetB0": {"baseline": 99.77, "distorted": 99.55, "wet_drop": 59.02, "gap": 40.75},
    "ShuffleNetV2":   {"baseline": 99.61, "distorted": 99.12, "wet_drop": 67.94, "gap": 31.67},
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available()
                       else "cpu")

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────
# MODEL BUILDING
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_model(model_name):
    num_classes = 38
    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ShuffleNetV2":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(model_name, setting, models_dir="models"):
    fname = f"{model_name}_{setting}_best.pth"
    path  = os.path.join(models_dir, fname)
    if not os.path.exists(path):
        return None, f"Model file not found: {path}"
    model = build_model(model_name)
    try:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# ──────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMG_MEAN, IMG_STD),
    ])
    return transform(image).unsqueeze(0)

# ──────────────────────────────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────────────────────────────
def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1).squeeze()
    top5_probs, top5_idx = torch.topk(probs, 5)
    return (
        CLASS_NAMES[probs.argmax().item()],
        probs.max().item() * 100,
        [(CLASS_NAMES[i], p.item() * 100) for i, p in zip(top5_idx, top5_probs)],
    )

# ──────────────────────────────────────────────────────────────────────
# GRAD-CAM
# ──────────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, model_name):
        self.model       = model
        self.gradients   = None
        self.activations = None

        if model_name == "MobileNetV2":
            layer = model.features[-1]
        elif model_name == "EfficientNetB0":
            layer = model.features[-1]
        elif model_name == "ShuffleNetV2":
            layer = model.conv5

        layer.register_forward_hook(self._save_act)
        layer.register_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def generate(self, tensor, class_idx=None):
        tensor = tensor.requires_grad_(True)
        output = self.model(tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * self.activations).sum(1, keepdim=True))
        cam     = cam.squeeze().cpu().numpy()
        cam     = cv2.resize(cam, (224, 224))
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

def overlay_gradcam(original_np, heatmap):
    h8  = np.uint8(255 * heatmap)
    col = cv2.applyColorMap(h8, cv2.COLORMAP_JET)
    col = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
    out = (original_np * 0.5 + col * 0.5).astype(np.uint8)
    return out

# ──────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/leaf-fluttering-in-wind.png", width=60)
    st.title("Crop Disease Detection")
    st.caption("MSc Capstone — University of Galway")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Predict", "📊 Results Dashboard"],
        index=0,
    )
    st.divider()
    st.caption("MobileNetV2 · EfficientNetB0 · ShuffleNetV2")
    st.caption("PlantVillage — 38 classes · 54,305 images")

# ──────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":

    st.title("🌿 Crop Disease Detection Under Water Droplet Distortions")
    st.markdown(
        "**MSc Capstone Project** · Aarti Bandgar · University of Galway · 2026"
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The Problem")
        st.write(
            "Deep learning models achieve above **99% accuracy** on clean leaf "
            "images but fail badly when leaves are wet in real field conditions. "
            "A farmer checking crops during or after rain is exactly when the model "
            "breaks down."
        )
        st.info(
            "Mohanty et al. (2016) showed models drop from **99.35% to 31%** "
            "accuracy on real field photographs.",
            icon="📉",
        )

    with col2:
        st.subheader("The Solution")
        st.write(
            "A physics-inspired synthetic water droplet augmentation pipeline "
            "renders realistic droplets onto clean leaf images. "
            "Three lightweight CNN models are trained under both clean and "
            "distorted conditions and evaluated for real-world robustness."
        )
        st.success(
            "Models trained on distorted images recover **above 99% accuracy** "
            "even on wet leaf test sets.",
            icon="✅",
        )

    st.divider()
    st.subheader("Key Finding — Wet-Condition Robustness Test")

    c1, c2, c3 = st.columns(3)
    for col, name, r in zip(
        [c1, c2, c3],
        ["MobileNetV2", "EfficientNetB0", "ShuffleNetV2"],
        [RESULTS["MobileNetV2"], RESULTS["EfficientNetB0"], RESULTS["ShuffleNetV2"]],
    ):
        with col:
            st.metric(
                label=f"{name}",
                value=f"{r['wet_drop']}%",
                delta=f"−{r['gap']}% on wet leaves",
                delta_color="inverse",
            )
            st.caption(f"Clean accuracy: {r['baseline']}%")

    st.warning(
        "When clean-trained models are tested on wet leaf images, all three models "
        "lose 31–41 percentage points of accuracy. "
        "MobileNetV2 goes from 44 wrong predictions to approximately 3,950 — "
        "**90× more errors** caused entirely by water droplets.",
        icon="⚠️",
    )

    st.divider()
    st.subheader("📄 Research Paper")
    
    paper_path = "Aarti_capstone_report.pdf"
    if os.path.exists(paper_path):
        with open(paper_path, "rb") as f:
            pdf_bytes = f.read()
        
        import base64
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        pdf_display = f'''
            <embed
                src="data:application/pdf;base64,{base64_pdf}"
                width="100%"
                height="800px"
                type="application/pdf"
            />
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.info("Paper PDF not found — add paper.pdf to the repo root.")
    
    
    st.divider()
    st.subheader("Pipeline Overview")
    steps = [
        ("1️⃣", "PlantVillage Dataset", "54,305 colour images · 38 disease classes"),
        ("2️⃣", "Synthetic Droplet Generation", "22 droplets · refraction · blur · alpha blend"),
        ("3️⃣", "Model Training", "Baseline (clean) and Distorted (droplet) settings"),
        ("4️⃣", "Robustness Evaluation", "Clean-trained model tested on wet images"),
        ("5️⃣", "Grad-CAM Analysis", "Verify model focuses on disease, not droplets"),
    ]
    cols = st.columns(5)
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)

# ──────────────────────────────────────────────────────────────────────
# PAGE: PREDICT
# ──────────────────────────────────────────────────────────────────────
elif page == "🔍 Predict":

    st.title("🔍 Disease Prediction")
    st.write(
        "Upload a leaf image to classify the disease and visualise "
        "where the model is looking using Grad-CAM."
    )

    col_cfg, col_main = st.columns([1, 2])

    with col_cfg:
        st.subheader("Settings")
        model_name = st.selectbox(
            "Model",
            ["MobileNetV2", "EfficientNetB0", "ShuffleNetV2"],
            help="All three are trained on PlantVillage 38 classes",
        )
        setting = st.selectbox(
            "Training setting",
            ["baseline", "distorted"],
            format_func=lambda x: "Baseline (clean trained)" if x == "baseline"
                                   else "Distorted (droplet trained)",
        )
        show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)

        st.info(
            f"**{model_name}**\n\n"
            f"Parameters: {MODEL_CONFIG[model_name]['params']}\n\n"
            f"Feature dim: {MODEL_CONFIG[model_name]['feature_dim']}"
        )

        uploaded = st.file_uploader(
            "Upload a leaf image",
            type=["jpg", "jpeg", "png"],
            help="Any PlantVillage-style leaf photograph",
        )

    with col_main:
        if uploaded is None:
            st.info("Upload a leaf image on the left to get started.", icon="👈")
            st.image(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
                caption="Upload a leaf photo — tomato, potato, apple, pepper, etc.",
                use_container_width=True,
            ) if False else None
        else:
            image = Image.open(uploaded).convert("RGB")
            image_np = np.array(image.resize((224, 224)))

            # Load model
            with st.spinner(f"Loading {model_name} ({setting})..."):
                model, err = load_model(model_name, setting)

            if err:
                st.error(
                    f"Could not load model: {err}\n\n"
                    "Make sure trained `.pth` files are in the `models/` folder. "
                    "Expected filename: `{model_name}_{setting}_best.pth`"
                )
            else:
                tensor = preprocess(image)

                # Prediction
                with st.spinner("Running prediction..."):
                    pred_class, confidence, top5 = predict(model, tensor)

                # Display
                st.subheader("Prediction")
                col_img, col_res = st.columns([1, 1])

                with col_img:
                    st.image(
                        image_np,
                        caption="Input image (224×224)",
                        use_container_width=True,
                    )

                with col_res:
                    if confidence > 90:
                        st.success(f"**{pred_class}**", icon="✅")
                    elif confidence > 70:
                        st.warning(f"**{pred_class}**", icon="⚠️")
                    else:
                        st.error(f"**{pred_class}**", icon="❓")

                    st.metric("Confidence", f"{confidence:.1f}%")

                    st.write("**Top 5 predictions:**")
                    for cls, prob in top5:
                        st.progress(prob / 100, text=f"{cls}: {prob:.1f}%")

                # Grad-CAM
                if show_gradcam:
                    st.divider()
                    st.subheader("Grad-CAM — Where the model looked")

                    with st.spinner("Generating Grad-CAM..."):
                        try:
                            gc      = GradCAM(model, model_name)
                            heatmap = gc.generate(tensor)
                            overlay = overlay_gradcam(image_np, heatmap)

                            gc_col1, gc_col2, gc_col3 = st.columns(3)
                            with gc_col1:
                                st.image(image_np, caption="Original", use_container_width=True)
                            with gc_col2:
                                hm_color = cv2.applyColorMap(
                                    np.uint8(255 * heatmap), cv2.COLORMAP_JET
                                )
                                st.image(
                                    cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB),
                                    caption="Heatmap (red = high attention)",
                                    use_container_width=True,
                                )
                            with gc_col3:
                                st.image(overlay, caption="Overlay", use_container_width=True)

                            st.caption(
                                "🔴 Red/yellow = model focused here strongly  "
                                "🔵 Blue/green = model ignored this area"
                            )
                        except Exception as e:
                            st.warning(f"Grad-CAM failed: {e}")

# ──────────────────────────────────────────────────────────────────────
# PAGE: RESULTS DASHBOARD
# ──────────────────────────────────────────────────────────────────────
elif page == "📊 Results Dashboard":

    st.title("📊 Experimental Results")
    st.write("All results from the trained MobileNetV2, EfficientNetB0, and ShuffleNetV2 models.")

    # Key metrics
    st.subheader("Test Set Accuracy — All Models and Settings")
    import pandas as pd

    df_results = pd.DataFrame({
        "Model":              ["MobileNetV2", "EfficientNetB0", "ShuffleNetV2"],
        "Baseline Acc (%)":   [99.59, 99.77, 99.61],
        "Distorted Acc (%)":  [99.37, 99.55, 99.12],
        "Macro F1 (Baseline)": [99.38, 99.70, 99.40],
        "Macro F1 (Distorted)": [99.24, 99.24, 98.91],
        "Gap (pp)":            [0.22, 0.22, 0.49],
    })
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    st.divider()

    # Robustness test
    st.subheader("Wet-Condition Robustness Evaluation")
    st.write(
        "Each **Baseline model** (trained on clean images) is evaluated on "
        "distorted test images to simulate real wet-leaf field conditions."
    )

    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1, c2, c3], ["MobileNetV2", "EfficientNetB0", "ShuffleNetV2"]):
        r = RESULTS[name]
        with col:
            st.markdown(f"### {name}")
            st.metric("On clean images", f"{r['baseline']}%")
            st.metric(
                "On wet leaf images",
                f"{r['wet_drop']}%",
                delta=f"−{r['gap']}%",
                delta_color="inverse",
            )

    # Bar chart
    st.divider()
    fig, ax = plt.subplots(figsize=(10, 5))
    models_list = ["MobileNetV2", "EfficientNetB0", "ShuffleNetV2"]
    x = range(len(models_list))
    w = 0.25
    bars1 = ax.bar([i - w for i in x],
                   [RESULTS[m]["baseline"] for m in models_list],
                   w, label="Baseline (clean)", color="#4CAF50", edgecolor="black", lw=0.7)
    bars2 = ax.bar(x,
                   [RESULTS[m]["distorted"] for m in models_list],
                   w, label="Distorted (wet trained)", color="#2196F3", edgecolor="black", lw=0.7)
    bars3 = ax.bar([i + w for i in x],
                   [RESULTS[m]["wet_drop"] for m in models_list],
                   w, label="Clean-trained on wet images", color="#F44336", edgecolor="black", lw=0.7)

    for bars in [bars1, bars2, bars3]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                    f"{b.get_height():.1f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(models_list, fontsize=11)
    ax.set_ylim(50, 107)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy Across All Models and Conditions", fontweight="bold")
    ax.axhline(99, color="gray", ls="--", lw=1, alpha=0.5, label="99% line")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Confusion matrix table
    st.subheader("Confusion Matrix Summary (summed across 38 classes)")
    df_cm = pd.DataFrame([
        {"Model": "MobileNetV2",    "Setting": "Baseline",  "TP": "10,815", "FP": 46, "FN": 46, "TN": "401,811"},
        {"Model": "MobileNetV2",    "Setting": "Distorted", "TP": "10,791", "FP": 70, "FN": 70, "TN": "401,787"},
        {"Model": "EfficientNetB0", "Setting": "Baseline",  "TP": "10,834", "FP": 27, "FN": 27, "TN": "401,830"},
        {"Model": "EfficientNetB0", "Setting": "Distorted", "TP": "10,813", "FP": 48, "FN": 48, "TN": "401,809"},
        {"Model": "ShuffleNetV2",   "Setting": "Baseline",  "TP": "10,817", "FP": 44, "FN": 44, "TN": "401,813"},
        {"Model": "ShuffleNetV2",   "Setting": "Distorted", "TP": "10,762", "FP": 99, "FN": 99, "TN": "401,758"},
    ])
    st.dataframe(df_cm, use_container_width=True, hide_index=True)
    st.caption(
        "TN values exceed 400,000 because TN is counted once per class per image. "
        "With 38 classes and 10,861 test images, every correctly rejected class "
        "adds one TN — this is expected in multi-class problems."
    )

    st.divider()

    # Comparison table
    st.subheader("Comparison with Existing Work")
    df_lit = pd.DataFrame([
        {"Reference": "Mohanty et al. (2016)", "Model": "AlexNet",               "Dataset": "PlantVillage (clean)",      "Accuracy": "85.53%", "Year": 2016},
        {"Reference": "Mohanty et al. (2016)", "Model": "GoogLeNet",             "Dataset": "PlantVillage (clean)",      "Accuracy": "99.34%", "Year": 2016},
        {"Reference": "Delnevo et al. (2021)", "Model": "MobileNetV2 + IoT",     "Dataset": "PlantVillage (clean)",      "Accuracy": "94.58%", "Year": 2021},
        {"Reference": "Garg et al. (2021)",    "Model": "MobileNetV2 (CROPCARE)","Dataset": "PlantVillage (clean)",      "Accuracy": "96.12%", "Year": 2021},
        {"Reference": "Iqbal (2025)",          "Model": "Transformer + CNN",     "Dataset": "PlantVillage (clean)",      "Accuracy": "96.20%", "Year": 2025},
        {"Reference": "This study",            "Model": "EfficientNetB0",        "Dataset": "PlantVillage (clean)",      "Accuracy": "99.77% ⭐", "Year": 2026},
        {"Reference": "Krishna et al. (2025)", "Model": "EfficientNet-B3",       "Dataset": "PlantDoc + web (field)",    "Accuracy": "80.19%", "Year": 2025},
        {"Reference": "This study",            "Model": "All 3 models (distorted)","Dataset": "PlantVillage (droplets)","Accuracy": "99.12–99.55%", "Year": 2026},
        {"Reference": "This study ⚠️",         "Model": "All 3 (clean-trained)", "Dataset": "Clean model on wet images","Accuracy": "59–68%", "Year": 2026},
    ])
    st.dataframe(df_lit, use_container_width=True, hide_index=True)

    st.info(
        "The wet-condition row (59–68%) has no equivalent in any prior study. "
        "This is the first published measurement of what happens when a "
        "clean-trained disease model encounters wet leaf images.",
        icon="📌",
    )
