import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
from PIL import Image
import pickle
from qdrant_client import QdrantClient
import os

# ----------------- QDRANT -----------------
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)

COLLECTION_NAME = "fashion_embeddings"

# ----------------- LOAD FILES -----------------
filenames = pickle.load(open("filenames.pkl", "rb"))

# ----------------- MODEL -----------------
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# ----------------- FEATURE EXTRACTION -----------------
def extract_features(img):
    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    feat = model.predict(arr).flatten()
    return feat / norm(feat)

# ----------------- UI -----------------
st.title("Fashion Recommendation (Local)")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img)

    query = extract_features(img)

    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query.tolist(),
        limit=5
    )

    st.subheader("Recommendations")
    for hit in hits:
        st.image(filenames[hit.id])
