# import os
# import io
# import base64
# from flask import Flask, render_template, request
# from tensorflow.keras.models import load_model
# from tensorflow.keras.losses import MeanSquaredError
# from PIL import Image, ImageChops, ImageEnhance
# import numpy as np
# import matplotlib.pyplot as plt

# app = Flask(__name__)

# # Define image size
# IMAGE_SIZE = (128, 128)

# def perform_ela(image, quality=90):
#     """
#     Perform Error Level Analysis (ELA) on the image.
#     Save the image at lower quality, compute the difference, and amplify it.
#     """
#     buffer = io.BytesIO()
#     image.save(buffer, 'JPEG', quality=quality)
#     buffer.seek(0)
#     compressed_image = Image.open(buffer)
#     ela_image = ImageChops.difference(image, compressed_image)
    
#     # Get the maximum difference value for scaling
#     extrema = ela_image.getextrema()
#     max_diff = max([ex[1] for ex in extrema])
#     if max_diff == 0:
#         max_diff = 1
#     scale = 255.0 / max_diff
#     ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
#     return ela_image

# def load_and_preprocess_image(image, target_size=IMAGE_SIZE):
#     """
#     Resize the image to target size and normalize the pixel values.
#     """
#     image_resized = image.resize(target_size)
#     return np.array(image_resized) / 255.0

# # Load the model with custom objects.
# custom_objects = {"mse": MeanSquaredError()}
# model = load_model("model1.h5", custom_objects=custom_objects)
# # model.summary()  # Optional: print model architecture in the console

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/general', methods=['GET', 'POST'])
# def general():
#     result_image = None
#     if request.method == 'POST':
#         file = request.files.get('image')
#         if file:
#             # Open the uploaded image
#             image = Image.open(file.stream).convert('RGB')
            
#             # Preprocess the original image (x1)
#             x1 = load_and_preprocess_image(image, IMAGE_SIZE)
            
#             # Generate and preprocess the ELA image (x2)
#             ela_image = perform_ela(image)
#             x2 = load_and_preprocess_image(ela_image, IMAGE_SIZE)
            
#             # Prepare model inputs: add batch dimension
#             x1_input = np.expand_dims(x1, axis=0)
#             x2_input = np.expand_dims(x2, axis=0)
            
#             # Get the prediction from the model (assumes two inputs)
#             prediction = model.predict([x1_input, x2_input])
#             # Assuming the predicted mask is in the first channel
#             predicted_mask = prediction[0, :, :, 0]
            
#             # Create a matplotlib figure with three subplots
#             fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#             axes[0].imshow(x1)
#             axes[0].set_title("Original")
#             axes[0].axis('off')
            
#             axes[1].imshow(x2)
#             axes[1].set_title("ELA")
#             axes[1].axis('off')
            
#             axes[2].imshow(predicted_mask, cmap='gray')
#             axes[2].set_title("Prediction")
#             axes[2].axis('off')
            
#             # Save the figure to a PNG image in memory and encode it as base64
#             buf = io.BytesIO()
#             plt.savefig(buf, format='png')
#             buf.seek(0)
#             result_image = base64.b64encode(buf.getvalue()).decode('utf-8')
#             plt.close(fig)
            
#     return render_template('general.html', result_image=result_image)

# if __name__ == '__main__':
#     app.run(debug=True)







import os
import io
import base64
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO  # For document analysis

app = Flask(__name__)

# Define image size for general analysis
IMAGE_SIZE = (128, 128)

#########################################
# General Image Analysis Functions & Model
#########################################
def perform_ela(image, quality=90):
    """
    Perform Error Level Analysis (ELA) on the image.
    Save the image at lower quality, compute the difference, and amplify it.
    """
    buffer = io.BytesIO()
    image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    ela_image = ImageChops.difference(image, compressed_image)
    
    # Get the maximum difference value for scaling
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

def load_and_preprocess_image(image, target_size=IMAGE_SIZE):
    """
    Resize the image to target size and normalize the pixel values.
    """
    image_resized = image.resize(target_size)
    return np.array(image_resized) / 255.0

# Load the Keras model for general image analysis.
custom_objects = {"mse": MeanSquaredError()}
general_model = load_model("model1.h5", custom_objects=custom_objects)
# general_model.summary()  # Optional: print model architecture

#########################################
# Document Analysis: YOLO Model
#########################################
# Load the YOLO model for document analysis.
document_model = YOLO("best1l.pt")  # Ensure the YOLO model file is available

#########################################
# Routes
#########################################
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/general', methods=['GET', 'POST'])
def general():
    result_image = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            # Open and process the uploaded image.
            image = Image.open(file.stream).convert('RGB')
            
            # Preprocess for the two branches.
            x1 = load_and_preprocess_image(image, IMAGE_SIZE)
            ela_image = perform_ela(image)
            x2 = load_and_preprocess_image(ela_image, IMAGE_SIZE)
            
            # Add batch dimension.
            x1_input = np.expand_dims(x1, axis=0)
            x2_input = np.expand_dims(x2, axis=0)
            
            # Predict the mask.
            prediction = general_model.predict([x1_input, x2_input])
            predicted_mask = prediction[0, :, :, 0]
            
            # Create a figure with three panels: Original, ELA, Prediction.
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(x1)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            axes[1].imshow(x2)
            axes[1].set_title("ELA")
            axes[1].axis('off')
            
            axes[2].imshow(predicted_mask, cmap='gray')
            axes[2].set_title("Prediction")
            axes[2].axis('off')
            
            # Save figure to PNG and encode in base64.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            result_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
    return render_template('general.html', result_image=result_image)

@app.route('/document', methods=['GET', 'POST'])
def document():
    result_image = None
    message = None
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            # Save the uploaded image to a temporary file.
            ext = file.filename.split('.')[-1]
            temp_filename = f"uploaded_doc.{ext}"
            file.save(temp_filename)
            
            # Run YOLO prediction on the document image.
            results = document_model.predict(source=temp_filename, save=True, conf=0.25)
            
            # Get the directory where YOLO saved the predictions.
            prediction_dir = results[0].save_dir
            predicted_image_path = os.path.join(prediction_dir, os.path.basename(temp_filename))
            
            # Load original and predicted images.
            original_image = Image.open(temp_filename).convert('RGB')
            predicted_image = Image.open(predicted_image_path).convert('RGB')
            
            # Create a figure with two panels: Original and Predicted.
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image")
            axes[0].axis('off')
            
            axes[1].imshow(predicted_image)
            axes[1].set_title("Predicted Image")
            axes[1].axis('off')
            
            # Save figure to PNG and encode in base64.
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            result_image = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            message = f"Prediction complete! Results saved in: {prediction_dir}"
            # Optionally, clean up the temporary file after processing.
            # os.remove(temp_filename)
            
    return render_template('document.html', result_image=result_image, message=message)

if __name__ == '__main__':
    app.run(debug=True)


