import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from PIL import Image
import pickle
import os
import gdown
from qdrant_client import QdrantClient

# ----------------- CHECK ENV VARIABLES -----------------
if "QDRANT_URL" not in os.environ or "QDRANT_API_KEY" not in os.environ:
    st.error("❌ Qdrant credentials not set in environment variables")
    st.stop()

# ----------------- QDRANT CLIENT -----------------
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

COLLECTION_NAME = "fashion_embeddings"

# ----------------- DOWNLOAD filenames.pkl -----------------
FILENAMES_URL = "https://drive.google.com/uc?id=1OKX-1ys4jqznVgX1e3cL1oOdLv0asaUY"

@st.cache_resource
def load_filenames():
    # Download if not exists
    if not os.path.exists("filenames.pkl"):
        gdown.download(FILENAMES_URL, "filenames.pkl", quiet=False)
    with open("filenames.pkl", "rb") as f:
        return pickle.load(f)

filenames = load_filenames()

# ----------------- MODEL -----------------
@st.cache_resource
def load_model():
    base = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base.trainable = False

    return tf.keras.Sequential([
        base,
        GlobalMaxPooling2D()
    ])

model = load_model()

# ----------------- FEATURE EXTRACTION -----------------
def extract_features(img: Image.Image):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    processed = preprocess_input(expanded)
    features = model.predict(processed, verbose=0).flatten()
    return features / norm(features)

# ----------------- UI -----------------
st.title("👕 Fashion Recommendation System")

uploaded_file = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Finding similar outfits..."):
        query_vector = extract_features(img)

        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector.tolist(),
            limit=5
        )

    st.subheader("✨ Recommended Items")
    cols = st.columns(5)
    for col, hit in zip(cols, results):
        with col:
            st.image(filenames[hit.id], use_column_width=True)
