import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("best1l.pt")  # Replace with the path to your trained YOLO model

# Streamlit UI
st.title("Forgery Detection with YOLO")
st.write("Upload an image to detect forgery and compare results side by side.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded image
    input_image_path = f"uploaded_image.{uploaded_file.name.split('.')[-1]}"
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.subheader("Uploaded Image")
    uploaded_image = Image.open(input_image_path)
    # st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    st.image(uploaded_image, caption="Uploaded Image")

    # Perform prediction
    st.subheader("Prediction")
    results = model.predict(source=input_image_path, save=True, conf=0.25)

    # Get the path of the saved prediction
    prediction_dir = results[0].save_dir
    predicted_image_path = os.path.join(prediction_dir, os.path.basename(input_image_path))

    # Display the prediction side by side
    col1, col2 = st.columns(2)
    with col1:
        # st.image(uploaded_image, caption="Original Image", use_container_width=True)
        st.image(uploaded_image, caption="Original Image")
    with col2:
        # st.image(predicted_image_path, caption="Predicted Image", use_container_width=True)
        st.image(predicted_image_path, caption="Predicted Image")

    st.success(f"Prediction complete! Results saved in: {prediction_dir}")
