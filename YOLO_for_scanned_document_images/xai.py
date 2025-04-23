import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


def preprocess_image(image_path, device):
    img = Image.open(image_path).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    
    img_tensor = preprocess(img).unsqueeze(0).to(device)  
    return img_tensor, img

if __name__ == '__main__':
    device = torch.device("cpu")
    model = YOLO("best1l.pt") 

    image_path = 'D:/ML-PROJECTS/Forgery2/train/images/1_jpg.rf.c6c28432e081900efdf0ad1973bdb900.jpg'  
    img_tensor, img = preprocess_image(image_path, device)

    model.eval()

    results = model(image_path)  


    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Extract the results
        boxes = results[0].boxes  
        xywh = boxes.xywh.cpu().numpy()  
        conf = boxes.conf.cpu().numpy() 
        cls = boxes.cls.cpu().numpy()  
        names = results[0].names  

        confidence_threshold = 0.2  
        valid_boxes = conf >= confidence_threshold 

        if valid_boxes.sum() > 0:
            img_np = np.array(img)
            
            for idx in range(len(valid_boxes)):
                if valid_boxes[idx]:
                    x1, y1, w, h = xywh[idx]
                    x1, y1 = int(x1 - w / 2), int(y1 - h / 2)  
                    x2, y2 = int(x1 + w), int(y1 + h)  
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_np, f'{names[int(cls[idx])]}: {conf[idx]:.2f}', 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    plt.imshow(img_np)
                    plt.axis('off')
                    plt.title(f"Prediction: {names[int(cls[idx])]} ({conf[idx]:.2f})")
                    plt.show()
                    img_tensor.requires_grad_()  
                    output = model(img_tensor)

                    if output[0].boxes is not None and len(output[0].boxes) > 0:
                        model.zero_grad()
                        output[0].boxes.conf[idx].backward()  
                        saliency = img_tensor.grad.data.abs().squeeze().cpu().numpy()  
                        saliency_map = np.maximum(saliency, 0)  
                        saliency_map = saliency_map / saliency_map.max()  
                        saliency_map = np.uint8(saliency_map * 255)  
                        plt.imshow(saliency_map, cmap='hot')
                        plt.axis('off')
                        plt.title(f"Saliency Map for {names[int(cls[idx])]} ({conf[idx]:.2f})")
                        plt.show()

                    else:
                        print(f"No valid output from the model for saliency map generation for {names[int(cls[idx])]}.")

        else:
            print("No valid detections above confidence threshold.")
    else:
        print("No objects detected in the image.")
