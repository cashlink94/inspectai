import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "6. Maintenance Department/resunet-segmentation-weights.hdf5"
IMG_SIZE = 256

# 🔥 More sensitive detection
CONF_THRESHOLD = 0.3
DEFECT_THRESHOLD = 0.001

# Colors per class (RGB)
CLASS_COLORS = {
    1: (255, 0, 0),     # Red
    2: (0, 255, 0),     # Green
    3: (0, 0, 255),     # Blue
    4: (255, 255, 0)    # Yellow
}

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ==============================
# UI
# ==============================
st.title("🧠 InspectAI")
st.markdown("### Automated Steel Defect Detection using Deep Learning")
st.warning("⚠️ Use steel surface images from the dataset for accurate results")

uploaded_file = st.file_uploader("📤 Upload Steel Image", type=["jpg", "jpeg", "png"])

# ==============================
# MAIN PIPELINE
# ==============================
if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=500)

    img = np.array(image)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img_norm = img_resized / 255.0

    # Model input
    img_input = np.expand_dims(img_norm, axis=(0, -1))

    # ==============================
    # PREDICTION
    # ==============================
    pred = model.predict(img_input)

    confidence_map = np.max(pred[0], axis=-1)
    mask = np.argmax(pred[0], axis=-1)

    # DEBUG (VERY IMPORTANT)
    st.write("Classes detected:", np.unique(mask))

    # ==============================
    # MASK PROCESSING
    # ==============================
    binary_mask = ((mask != 0) & (confidence_map > CONF_THRESHOLD)).astype(np.uint8)

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # ==============================
    # VISUALIZATION (STRONG)
    # ==============================
    display_img = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    display_img = cv2.equalizeHist(display_img)
    original_display = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)

    overlay = original_display.copy()
    has_defect = False

    for class_id, color in CLASS_COLORS.items():
        class_pixels = ((mask == class_id) & (confidence_map > CONF_THRESHOLD))
        if np.any(class_pixels):
            has_defect = True
            overlay[class_pixels] = color

    # Blend only if defects exist
    if has_defect:
        blended = cv2.addWeighted(original_display, 0.7, overlay, 0.3, 0)
    else:
        blended = original_display.copy()

    # ==============================
    # BOUNDING BOXES
    # ==============================
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_output = blended.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(final_output, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # ==============================
    # METRICS
    # ==============================
    defect_ratio = np.sum(binary_mask) / binary_mask.size

    st.markdown("---")
    st.subheader("📊 Detection Summary")

    if defect_ratio > DEFECT_THRESHOLD:
        st.error("❌ Defect Detected")
    else:
        st.success("✅ No Defect Detected")

    st.metric("Defect Coverage", f"{defect_ratio:.2%}")

    if defect_ratio == 0:
        st.info("ℹ️ Model detected a clean steel surface (no visible defects).")

    # ==============================
    # DISPLAY RESULTS
    # ==============================
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_display, caption="Original", width=500)

    with col2:
        st.image(final_output, caption="Detected Defects (Colors + Boxes)", width=500)

    st.image(binary_mask * 255, caption="Final Clean Mask", width=500)

    # ==============================
    # DEBUG MASK VIEW
    # ==============================
    mask_debug = mask.copy()
    mask_debug[mask_debug == 0] = 255
    st.image(mask_debug, caption="Debug Mask (White = Background)", width=500)