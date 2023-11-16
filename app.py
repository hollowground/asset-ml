import streamlit as st
import tensorflow as tf
import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
from fuzzywuzzy import fuzz
from PIL import Image
import tempfile
import keras_ocr
import pandas as pd

# MODEL_PATH = "models/multi_class_asset_class.keras"
# List to store folder names
MODEL_CLASSES = []
input_file = "cat_names.txt"
with open(input_file, "r") as file:
    MODEL_CLASSES = file.read().splitlines()

# MODEL_CLASSES = sorted(os.listdir("dataset"))
# MODEL_CLASSES = os.listdir("dataset")
st.set_option("deprecation.showfileUploaderEncoding", False)


@st.cache_resource
def load_ml_model():
    model = load_model(os.path.join("models", "multi_class_asset_class.h5"))
    return model


model = load_ml_model()

st.write(
    """
         # Manufacturer/Model Identification
         """
)

file = st.file_uploader("Please upload an equipment image", type=["jpg", "png"])


def get_matches(existing_text="", additional_text=None):
    matches = existing_text

    if additional_text:
        matches += additional_text + "\n"

    return matches


# Use OCR to find close matches
def extract_text_from_image(image_path, target_size=(400, 400)):
    # Open and resize the image
    original_image = image_path  # Image.open(image_path)
    resized_image = original_image.resize(target_size)

    # Save the resized image to a temporary file
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    resized_image.save(temp_file_path)

    try:
        # Create a pipeline
        pipeline = keras_ocr.pipeline.Pipeline()

        # Get predictions
        prediction_groups = pipeline.recognize([temp_file_path])

        # Create a DataFrame from predictions
        df = pd.DataFrame(prediction_groups[0], columns=["text", "bbox"])

        # Extract text from predictions
        extracted_text = " ".join([str(word_info) for word_info in df["text"]])

        return extracted_text
    finally:
        # Clean up: Remove the temporary file
        os.remove(temp_file_path)


def find_close_matches(query, string_list, top_n=5, threshold=10):
    # Use fuzzywuzzy to find close matches
    matches = [(s, fuzz.ratio(query, s)) for s in string_list]

    # Sort matches by score in descending order
    sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)

    # Take the top N matches
    top_matches = sorted_matches[:top_n]

    # Filter matches based on the threshold
    close_matches = [(s, score) for s, score in top_matches if score >= threshold]

    return close_matches


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

    # Extract text from the image
    extracted_text = extract_text_from_image(image_data)
    # Find and return the top 5 close matches in the list
    close_matches = find_close_matches(extracted_text, MODEL_CLASSES, top_n=5)
    predictions = get_matches(f"The predicted class is: {predicted_label}")
    if close_matches:
        predictions = get_matches(
            predictions, additional_text="\nTop 5 close text matches found:"
        )
        for match, score in close_matches:
            predictions = get_matches(
                predictions, additional_text=f"{match} (Score: {score})"
            )
    else:
        predictions = get_matches(
            predictions, additional_text=("No close matches found.")
        )
    print(predictions)
    return predictions


if file is None:
    st.text("Please upload an image...")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    if prediction is not None:
        textsplit = prediction.splitlines()
        for x in textsplit:
            st.success(x)  # st.write(x)
