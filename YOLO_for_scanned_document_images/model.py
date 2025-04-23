# from ultralytics import YOLO

# # yolo model creation
# model = YOLO("yolo-weights/yolo11l.pt")
# model.train(data="data.yaml", imgsz=320, batch=4, epochs=1, workers=4)

if __name__ == '__main__':
    # import ultralytics
    from ultralytics import YOLO

    # model = YOLO('yolov11l.pt')
    # # Load your model
    model = YOLO("yolo-weights/yolov8l.pt")
    model.train(data="data.yaml", imgsz=320, batch=12, epochs=100, lr0=0.001,workers=4)
    # model.train(
    #     data="data.yaml",  # Path to your dataset configuration file
    #     imgsz=640,         # Image size
    #     batch=64,           # Batch size
    #     epochs=25,          # Number of epochs
    #     workers=0          # Number of worker processes
    # )
