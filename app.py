import os
import streamlit as st
import numpy as np
from PIL import Image
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ---------------- CONFIG ----------------
COLLECTION_NAME = "fashion_embeddings"
VECTOR_SIZE = 2048

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    st.error("❌ Qdrant credentials not set in environment variables")
    st.stop()

# ---------------- QDRANT CLIENT ----------------
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)

# ---------------- MODEL (CACHED) ----------------
@st.cache_resource
def load_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    return tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

model = load_model()

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(img: Image.Image):
    img = img.convert("RGB")           # 🔥 FIX: RGBA → RGB
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0).flatten()
    return features / norm(features)

# ---------------- UI ----------------
st.set_page_config(page_title="Fashion Recommendation", layout="wide")
st.title("👕 Fashion Recommendation System")

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    with st.spinner("🔍 Finding similar outfits..."):
        query_vector = extract_features(img)

        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=5
        )

    st.subheader("Recommended Items")

    cols = st.columns(5)
    for col, point in zip(cols, search_result):
        image_url = point.payload.get("filename")
        if image_url:
            col.image(image_url, use_container_width=True)
