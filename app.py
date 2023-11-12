import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
import os

MODEL_PATH = "models/multi_class_asset_class.keras"
MODEL_CLASSES = sorted(os.listdir("dataset"))
# MODEL_CLASSES = os.listdir("dataset")
st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = load_model()

st.write(
    """
         # Manufacturer/Model Identification
         """
)

file = st.file_uploader("Please upload an equipment image", type=["jpg", "png"])


def import_and_predict(image_data, model):
    img = np.array(image_data)

    # Convert image to RGB (in case it has an alpha channel)
    img = Image.fromarray(img)
    img = img.convert("RGB")
    img = np.array(img)

    resized_img = cv2.resize(img, (256, 256))
    input_img = np.expand_dims(resized_img, axis=0)

    predicted_probs = model.predict(input_img / 255.0)  # Normalize the image data
    predicted_class = np.argmax(predicted_probs)
    predicted_label = MODEL_CLASSES[predicted_class]

    return f"The predicted class is: {predicted_label}"


if file is None:
    st.text("Please upload an image...")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    st.success(prediction)
