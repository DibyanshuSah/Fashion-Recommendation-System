import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle
import os
import gdown
from sklearn.neighbors import NearestNeighbors
from PIL import Image

# ----------------- DOWNLOAD FROM DRIVE -----------------
EMB_URL = "https://drive.google.com/uc?id=1lL-OIgrVNG7e2tDKT817bDGtDgYvIJC3"
FIL_URL = "https://drive.google.com/uc?id=1OKX-1ys4jqznVgX1e3cL1oOdLv0asaUY"

def download_if_needed():
    if not os.path.exists("embeddings.pkl"):
        gdown.download(EMB_URL, "embeddings.pkl", quiet=False)
    if not os.path.exists("filenames.pkl"):
        gdown.download(FIL_URL, "filenames.pkl", quiet=False)

download_if_needed()

# ----------------- LOAD DATA -----------------
feature_list = pickle.load(open("embeddings.pkl", "rb"))
filenames = pickle.load(open("filenames.pkl", "rb"))

# ----------------- MODEL -----------------
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# ----------------- FEATURE EXTRACT -----------------
def extract_features(img, model):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    processed = preprocess_input(expanded)
    result = model.predict(processed, verbose=0).flatten()
    return result / norm(result)

# ----------------- UI -----------------
st.title("Fashion Recommendation System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg","webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    query = extract_features(img, model)

    neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([query])

    st.subheader("Recommended Items")
    for idx in indices[0]:
        st.image(filenames[idx], use_column_width=True)
