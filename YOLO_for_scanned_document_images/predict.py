from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("D:/ML-PROJECTS/Forgery2/runs/detect/train/weights/best.pt")  # Replace with the path to your `best.pt` file

# Path to the input image
image_path = "path/to/your/image.jpg"  # Replace with the path to your input image

# Perform inference
results = model(image_path)

# Display results
# Save results to an image file (optional)
results.save("predictions/")  # Saves the results in the 'predictions/' directory

# Load the image with predictions to display
predicted_image_path = "predictions/image0.jpg"  # Change the name if needed
predicted_image = cv2.imread(predicted_image_path)

# Convert to RGB for display using matplotlib
predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)

# Display the image
plt.imshow(predicted_image)
plt.axis("off")
plt.show()

# Print detection results
for result in results:
    print(f"Detected objects: {result.boxes.data}")
