import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ========= CONFIG =========
MODEL_PATH = "inspectai_model.pth"
IMG_SIZE = 128
DEVICE = "cpu"

CLASS_NAMES = ["Defect 1", "Defect 2", "Defect 3", "Defect 4"]


# ========= LOAD MODEL =========
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    return model


# ========= TRANSFORM =========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


# ========= PREDICT =========
def predict(image, model):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs[0][pred].item()


# ========= UI =========
st.title("🧠 InspectAI")
st.write("Automated Steel Defect Detection")

uploaded_file = st.file_uploader("📤 Upload Steel Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    model = load_model()

    pred, confidence = predict(image, model)

    st.success(f"Prediction: {CLASS_NAMES[pred]}")
    st.write(f"Confidence: {confidence:.2f}")