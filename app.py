import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from PIL import Image
from qdrant_client import QdrantClient
import os

# ---------------- CONFIG (ENV VARS) ----------------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "fashion_embeddings"

if QDRANT_URL is None or QDRANT_API_KEY is None:
    st.error("❌ Qdrant credentials not set in environment variables")
    st.stop()

# ---------------- QDRANT CLIENT ----------------
@st.cache_resource
def get_qdrant():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60
    )

qdrant = get_qdrant()

# ---------------- MODEL ----------------
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    processed = preprocess_input(expanded)
    result = model.predict(processed, verbose=0).flatten()
    return result / norm(result)

# ---------------- UI ----------------
st.set_page_config(page_title="Fashion Recommender", layout="centered")
st.title("👕 Fashion Recommendation System")

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "png", "jpeg", "webp"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    query_vector = extract_features(img, model)

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector.tolist(),
        limit=5
    )

    st.subheader("Recommended Items")
    for hit in hits:
        st.image(hit.payload["filename"], use_column_width=True)
