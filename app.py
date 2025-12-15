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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fashion Recommender",
    layout="centered"
)

# ---------------- CONFIG ----------------
EMB_URL = "https://drive.google.com/uc?id=1lL-OIgrVNG7e2tDKT817bDGtDgYvIJC3"
FIL_URL = "https://drive.google.com/uc?id=1OKX-1ys4jqznVgX1e3cL1oOdLv0asaUY"

EMB_FILE = "embeddings.pkl"
FIL_FILE = "filenames.pkl"

# ---------------- DOWNLOAD FILES (ONCE) ----------------
@st.cache_resource(show_spinner=True)
def download_files():
    if not os.path.exists(EMB_FILE):
        gdown.download(EMB_URL, EMB_FILE, quiet=True)
    if not os.path.exists(FIL_FILE):
        gdown.download(FIL_URL, FIL_FILE, quiet=True)

download_files()

# ---------------- LOAD DATA ----------------
@st.cache_resource(show_spinner=False)
def load_data():
    with open(EMB_FILE, "rb") as f:
        embeddings = pickle.load(f)
    with open(FIL_FILE, "rb") as f:
        filenames = pickle.load(f)

    embeddings = np.array(embeddings).astype("float32")
    return embeddings, filenames

feature_list, filenames = load_data()

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=False)
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

# ---------------- NEAREST NEIGHBOR (CRITICAL FIX) ----------------
@st.cache_resource(show_spinner=False)
def load_nn_model(features):
    nn = NearestNeighbors(
        n_neighbors=5,
        metric="euclidean",
        algorithm="brute"   # 🔥 HF CPU SAFE
    )
    nn.fit(features)
    return nn

nn_model = load_nn_model(feature_list)

# ---------------- FEATURE EXTRACT ----------------
def extract_features(img):
    img = img.convert("RGB")        # 🔥 RGBA → RGB FIX
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    features = model.predict(arr, verbose=0).flatten()
    features = features / norm(features)
    return features.astype("float32")

# ---------------- UI ----------------
st.title("👕 Fashion Recommendation System")
st.write("Upload a fashion image to find similar products")

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    try:
        img = Image.open(uploaded)
        st.image(img, width=300)

        with st.spinner("Finding similar outfits..."):
            query = extract_features(img)
            _, indices = nn_model.kneighbors([query])

        st.subheader("Recommended Items")
        cols = st.columns(5)
        for col, idx in zip(cols, indices[0]):
            col.image(filenames[idx], width=150)

    except Exception as e:
        st.error("❌ Something went wrong")
        st.exception(e)
else:
    st.info("Please upload an image to continue")
