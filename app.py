import streamlit as st
import numpy as np
import pickle
import os
import gdown
from PIL import Image
from numpy.linalg import norm

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors

# ---------------- CONFIG ----------------
EMB_URL = "https://drive.google.com/uc?id=1lL-OIgrVNG7e2tDKT817bDGtDgYvIJC3"
FIL_URL = "https://drive.google.com/uc?id=1OKX-1ys4jqznVgX1e3cL1oOdLv0asaUY"

# ---------------- DOWNLOAD FILES ----------------
@st.cache_resource
def download_files():
    if not os.path.exists("embeddings.pkl"):
        gdown.download(EMB_URL, "embeddings.pkl", quiet=False)
    if not os.path.exists("filenames.pkl"):
        gdown.download(FIL_URL, "filenames.pkl", quiet=False)

download_files()

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_data():
    with open("embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    with open("filenames.pkl", "rb") as f:
        filenames = pickle.load(f)
    return embeddings, filenames

feature_list, filenames = load_data()

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base.trainable = False
    model = tf.keras.Sequential([
        base,
        GlobalAveragePooling2D()
    ])
    return model

model = load_model()

# ---------------- FEATURE EXTRACT ----------------
def extract_features(img):
    img = img.convert("RGB")        # 🔥 RGBA FIX
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    features = model.predict(arr, verbose=0).flatten()
    return features / norm(features)

# ---------------- UI ----------------
st.title("👕 Fashion Recommendation System")

uploaded = st.file_uploader(
    "Upload a fashion image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    img = Image.open(uploaded)
    st.image(img, width=300)

    with st.spinner("Finding similar outfits..."):
        query = extract_features(img)

        nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
        nn.fit(feature_list)
        _, indices = nn.kneighbors([query])

    st.subheader("Recommended Items")
    cols = st.columns(5)
    for col, idx in zip(cols, indices[0]):
        col.image(filenames[idx], width=150)
