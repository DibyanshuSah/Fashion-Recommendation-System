import pickle
import numpy as np
from PIL import Image
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

# LOAD DATA
feature_list = np.array(pickle.load(open("embeddings.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

# MODEL
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

def extract_features(img):
    img = img.convert("RGB")
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array).flatten()
    return features / norm(features)

# TEST
img = Image.open("test.jpg")
query = extract_features(img)

neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([query])
print(indices)
