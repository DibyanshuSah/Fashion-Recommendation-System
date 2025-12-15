import streamlit as st
import numpy as np
import pickle
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
    page_title="Fashion Outfit Recommender",
    layout="centered"
)

# ---------------- LOAD DATA (ONCE) ----------------
@st.cache_resource
def load_data():
    features = np.array(
        pickle.load(open("embeddings.pkl", "rb")),
        dtype="float32"
    )
    filenames = pickle.load(open("filenames.pkl", "rb"))
    return features, filenames

feature_list, filenames = load_data()

# ---------------- LOAD MODEL (ONCE) ----------------
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

# ---------------- LOAD NN (ONCE) ----------------
@st.cache_resource
def load_nn(features):
    nn = NearestNeighbors(
        n_neighbors=5,
        metric="euclidean",
        algorithm="brute"   # HF CPU SAFE
    )
    nn.fit(features)
    return nn

nn_model = load_nn(feature_list)

# ---------------- FEATURE EXTRACT ----------------
def extract_features(img):
    img = img.convert("RGB")        # RGBA → RGB FIX
    img = img.resize((224, 224))

    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    features = model.predict(arr, verbose=0).flatten()
    features = features / norm(features)
    return features.astype("float32")

# ---------------- UI ----------------
st.title("👕 Fashion Outfit Recommender")

uploaded = st.file_uploader(
    "Upload a fashion image",
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
