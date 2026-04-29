"""
Standalone Streamlit App for Gas Cylinder Classification
"""
import streamlit as st
import numpy as np
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms

# --- Constants ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
IMG_SIZE = 224

BRAND_CLASSES = ['HP', 'IND', 'NONE']
SIZE_CLASSES = ['2KG', '5KG', '15KG', 'NONE']
COLOR_CLASSES = ['BLUE', 'RED', 'NONE']
PRESENCE_CLASSES = ['PRESENT', 'ABSENT']

# --- Model Architecture ---
class GasCylinderDetector(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=False,
                 num_brands=3, num_sizes=4, num_colors=3, num_presence=2, dropout_rate=0.5):
        super(GasCylinderDetector, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feature_dim = 1280
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.brand_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2), nn.Linear(256, num_brands),
        )
        self.size_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2), nn.Linear(256, num_sizes),
        )
        self.color_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2), nn.Linear(256, num_colors),
        )
        self.presence_head = nn.Sequential(
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2), nn.Linear(128, num_presence),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        shared_features = self.shared_fc(features)
        return {
            'brand': self.brand_head(shared_features),
            'size': self.size_head(shared_features),
            'color': self.color_head(shared_features),
            'presence': self.presence_head(shared_features),
        }

# --- Load Model ---
@st.cache_resource
def get_model():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GasCylinderDetector(backbone_name='efficientnet_b0', pretrained=False)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device, None
    except Exception as e:
        return None, None, str(e)

def preprocess_image(image: Image.Image, device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(device)

# --- App UI ---
st.set_page_config(page_title="Gas Cylinder Classifier", layout="centered")

st.title("Gas Cylinder Classifier")
st.markdown("Upload an image of a gas cylinder to classify its brand, size, color, and presence.")

model, device, load_error = get_model()

if load_error:
    st.error(f"Failed to load model from {MODEL_PATH}")
    st.error(str(load_error))
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Classifying..."):
        input_tensor = preprocess_image(image, device)
        with torch.no_grad():
            preds = model(input_tensor)
        
        brand_probs = F.softmax(preds['brand'][0], dim=0).cpu().numpy()
        size_probs = F.softmax(preds['size'][0], dim=0).cpu().numpy()
        color_probs = F.softmax(preds['color'][0], dim=0).cpu().numpy()
        presence_probs = F.softmax(preds['presence'][0], dim=0).cpu().numpy()

        brand_idx = int(np.argmax(brand_probs))
        size_idx = int(np.argmax(size_probs))
        color_idx = int(np.argmax(color_probs))
        presence_idx = int(np.argmax(presence_probs))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Brand", BRAND_CLASSES[brand_idx], f"{brand_probs[brand_idx]*100:.1f}%")
            st.metric("Color", COLOR_CLASSES[color_idx], f"{color_probs[color_idx]*100:.1f}%")
        with col2:
            st.metric("Size", SIZE_CLASSES[size_idx], f"{size_probs[size_idx]*100:.1f}%")
            st.metric("Presence", PRESENCE_CLASSES[presence_idx], f"{presence_probs[presence_idx]*100:.1f}%")

        st.markdown("---")
        st.write("### Detailed Probabilities")
        
        def show_probs(task_name, probs, classes):
            st.write(f"**{task_name}**")
            for prob, cls in zip(probs, classes):
                st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

        c1, c2, c3, c4 = st.columns(4)
        with c1: show_probs("Brand", brand_probs, BRAND_CLASSES)
        with c2: show_probs("Size", size_probs, SIZE_CLASSES)
        with c3: show_probs("Color", color_probs, COLOR_CLASSES)
        with c4: show_probs("Presence", presence_probs, PRESENCE_CLASSES)
