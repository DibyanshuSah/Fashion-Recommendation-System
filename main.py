import streamlit as st
import numpy as np
import pickle
import os
from PIL import Image
from numpy.linalg import norm

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors

# ---------------- LOAD DATA ----------------
feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

# ---------------- MODEL ----------------
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

# ---------------- FEATURE EXTRACT ----------------
def extract_features(img_path, model):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    features = model.predict(arr).flatten()
    return features / norm(features)

# ---------------- UI ----------------
st.title("👕 Fashion Outfit Recommender (Local)")

uploaded = st.file_uploader("Upload an image")

if uploaded:
    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", uploaded.name)

    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.image(Image.open(uploaded), width=300)

    features = extract_features(path, model)

    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(feature_list)
    _, indices = nn.kneighbors([features])

    st.subheader("Recommendations")
    cols = st.columns(5)
    for col, idx in zip(cols, indices[0]):
        col.image(filenames[idx], width=150)
