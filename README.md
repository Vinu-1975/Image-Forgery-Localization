# Image Forgery Localization

This repository provides a comprehensive solution for detecting and localizing image forgeries, with a special focus on scanned document images. It combines two main approaches:

- **General Image Forgery Detection using ELA-CNN**: Uses Error Level Analysis (ELA) and a Convolutional Neural Network to localize manipulated regions in general images.
- **Document Forgery Detection using YOLO**: Utilizes a YOLO-based object detection model to identify forged regions in scanned documents.

---

## Features

- **Web Demo (Flask App)**: Upload images and visualize forgery localization results for both general images and scanned documents.
- **ELA-CNN Model**: Detects forgeries in general images using ELA preprocessing and a custom-trained CNN.
- **YOLO Model**: Detects forgeries in scanned documents using a YOLOv8-based model.
- **Visualization**: Side-by-side comparison of original, ELA, and predicted mask for general images; original and YOLO-predicted images for documents.
- **XAI Support**: Includes scripts for explainable AI (XAI) visualizations on YOLO predictions.

---

## Directory Structure

- `Demo Website/` : Flask web application for demo.
- `YOLO_for_scanned_document_images/` : YOLO training, prediction, and utility scripts for document forgery detection.
- `ELA_CNN_for_general_images/` : Scripts and notebooks for ELA-CNN model.
- `static/`, `templates/` : Assets and HTML templates for the web app.
- `best1l.pt`, `model1.h5` : Pretrained YOLO and ELA-CNN models (ensure these files are present for inference).

---

## Setup & Installation

1. **Clone the repository**
2. **Install dependencies**

   - Python 3.8+
   - Install required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Additional requirements:
     - `ultralytics`, `opencv-python`, `flask`, `tensorflow`, `Pillow`, `matplotlib`, `numpy`

3. **Download/Place Models**
   - Place your trained YOLO model (`best1l.pt`) in the root directory.
   - The ELA-CNN model file (`model1.h5`) is **not included** in this repository due to its large size (1.5 GB).  
     You can download it from [this Google Drive link](https://drive.google.com/file/d/17x9k8YKdSVZ3p6J3na21L1SEbnjHWzYp/view?usp=drive_link) and place it in the (`Demo Website`) directory.

---

## Usage

### 1. Run the Flask Web Demo

```bash
python Demo Website/app.py
```
