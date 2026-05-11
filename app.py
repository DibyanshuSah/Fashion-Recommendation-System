import gradio as gr
import numpy as np
import os
from PIL import Image
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel('ERROR')

IMAGE_DIR = "images"
image_files = os.listdir(IMAGE_DIR)

#model loading 
model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)
model.trainable = False

#feature extraction 
def extract_features(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    features = model.predict(img_array, verbose=0).flatten()
    return features / norm(features)

#logic
def recommend(uploaded_img):
    if uploaded_img is None:
        return []

    query = extract_features(uploaded_img)
    similarities = []

    for img_name in image_files:
        img_path = os.path.join(IMAGE_DIR, img_name)

        try:
            img = Image.open(img_path)
            feat = extract_features(img)

            sim = np.dot(query, feat.T)
            similarities.append((sim, img_path))
        except:
            continue

    similarities.sort(reverse=True)

    results = []
    for i in range(min(3, len(similarities))): 
        results.append(Image.open(similarities[i][1]))

    return results

#streamlit ui
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 👕 Fashion Recommender")
    gr.Markdown("Upload a fashion image and get similar product recommendations")

    with gr.Row():
        input_img = gr.Image(type="pil", label="Upload Fashion Image")
        output_imgs = gr.Gallery(
            label="Recommended Items",
            columns=3,
            rows=1,
            height="auto",        
            object_fit="cover",  
            preview=True      
        )
    with gr.Row():
        submit = gr.Button("Submit", variant="primary")
        clear = gr.Button("Clear")

    submit.click(fn=recommend, inputs=input_img, outputs=output_imgs)
    clear.click(lambda: (None, []), outputs=[input_img, output_imgs])
demo.launch()
