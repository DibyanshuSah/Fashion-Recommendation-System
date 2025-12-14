# Local-only version (not for deployment)

import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load local files (DO NOT COMMIT PKL FILES)
feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def extract_features(img, model):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    processed = preprocess_input(expanded)
    result = model.predict(processed, verbose=0).flatten()
    return result / norm(result)

st.title("Fashion Recommender (Local)")

uploaded_file = st.file_uploader("Upload image")

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img)

    query = extract_features(img, model)

    nn = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    nn.fit(feature_list)

    _, indices = nn.kneighbors([query])

    for idx in indices[0]:
        st.image(filenames[idx], width=150)
