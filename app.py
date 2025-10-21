import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title(" CIFAR-10 Image Classifier")

# CIFAR-10 labels
classes = ["plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# --- Load Model ---
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)  # use "weights=None" instead of deprecated "pretrained"
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10)
    )
    model.load_state_dict(torch.load(r"resnet50_cifar10_state.pth", map_location="cpu"))  # ✅ your file
    model.eval()
    return model

model = load_model()
st.write("upload image of plane, car, bird, cat, deer,dog, frog, horse, ship,truck to get prediction")
# --- Upload Section ---
uploaded = st.file_uploader(" Upload an image (jpg/png):", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Predict"):
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
        img_t = transform(img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(img_t)
            probabilities = torch.softmax(outputs, dim=1)[0]
            pred_class = torch.argmax(probabilities).item()
            confidence = probabilities[pred_class].item() * 100

        st.success(f" Prediction: **{classes[pred_class]}** ({confidence:.2f}% confidence)")
else:
    st.info("Please upload an image to start prediction.")
