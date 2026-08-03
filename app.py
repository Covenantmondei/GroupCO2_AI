import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Tomato Leaf Disease Classifier",
    page_icon="🍅",
    layout="wide"
)

# Custom Theme CSS — sleek dark palette to stand out from default Streamlit styling
st.markdown("""
<style>
    /* Viewport & Container Reset */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 98% !important;
    }
    
    /* App background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Header Card - Full Width */
    .header-box {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 1px solid #312e81;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 22px;
        width: 100%;
        box-sizing: border-box;
    }
    .header-box h1 {
        color: #818cf8;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
    }
    .header-box p {
        color: #94a3b8;
        margin: 6px 0 0 0;
        font-size: 0.95rem;
    }
    
    /* Custom Progress Bars */
    .bar-row {
        display: flex;
        justify-content: space-between;
        font-weight: 600;
        font-size: 0.9rem;
        color: #cbd5e1;
        margin-top: 10px;
    }
    .bar-bg {
        background: #1e293b;
        border-radius: 6px;
        height: 18px;
        width: 100%;
        overflow: hidden;
        margin-top: 4px;
        margin-bottom: 10px;
    }
    .bar-fill-mosaic {
        background: linear-gradient(90deg, #f59e0b, #ef4444);
        height: 100%;
    }
    .bar-fill-healthy {
        background: linear-gradient(90deg, #10b981, #06b6d4);
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown("""
<div class="header-box">
    <h1>🍅 Tomato Leaf Disease Classifier</h1>
    <p>Upload a tomato leaf image to check for <b>Tomato Mosaic Virus</b>.</p>
</div>
""", unsafe_allow_html=True)

CLASS_NAMES = ["Tomato Mosaic Virus", "Healthy"]
IMAGE_SIZE = (224, 224)

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/mobilenetv3_final.keras")
    return model

# Load "not a leaf" guard components
@st.cache_resource
def load_guard(_model):
    centroid = np.load("models/leaf_centroid.npy")
    threshold = float(np.load("models/leaf_threshold.npy")[0])

    pooling_layer = None
    for layer in _model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            pooling_layer = layer
            break

    embedding_model = tf.keras.Model(
        inputs=_model.input,
        outputs=pooling_layer.output
    )
    return embedding_model, centroid, threshold

def is_valid_leaf(pil_image, embedding_model, centroid, threshold):
    """Check if the image looks like a tomato leaf using embedding distance."""
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    emb = embedding_model.predict(arr, verbose=0)[0]
    distance = np.linalg.norm(emb - centroid)
    return distance <= threshold, distance

def predict(model, pil_image):
    """Preprocess image and return prediction label + confidence."""
    img = pil_image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    prob_mosaic = float(probs[0])
    prob_healthy = float(probs[1])

    return label, confidence, prob_healthy, prob_mosaic

# Main Layout

model = load_model()
embedding_model, centroid, threshold = load_guard(model)

# 2-Column Layout to fit everything on 1 page
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width="stretch", caption="Uploaded Image")
    else:
        st.info("Please upload an image to get a prediction.")

with col2:
    if uploaded_file:
        with st.spinner("Analyzing..."):
            valid, distance = is_valid_leaf(img, embedding_model, centroid, threshold)

        if not valid:
            st.error("🚫 This doesn't look like a tomato leaf. Please upload a clear tomato leaf image.")
        else:
            label, confidence, prob_healthy, prob_mosaic = predict(model, img)

            st.markdown(f"### Prediction: **{label}**")
            st.write(f"Confidence: **{confidence * 100:.1f}%**")

            st.markdown(f"""
            <div class="bar-row">
                <span>Healthy</span>
                <span>{prob_healthy * 100:.1f}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill-healthy" style="width: {prob_healthy * 100:.1f}%;"></div>
            </div>
            
            <div class="bar-row">
                <span>Tomato Mosaic Virus</span>
                <span>{prob_mosaic * 100:.1f}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill-mosaic" style="width: {prob_mosaic * 100:.1f}%;"></div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            if confidence < 0.60:
                st.warning("⚠️ Low confidence prediction — result may be unreliable.")

            if label == "Tomato Mosaic Virus":
                st.warning("⚠️ This leaf shows signs of Tomato Mosaic Virus. Consider consulting an agricultural expert.")
            else:
                st.success("✅ This leaf appears healthy.")