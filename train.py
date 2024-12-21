from ultralytics import YOLO

# Load a pre-trained YOLOv11 model
model = YOLO('yolo11n.pt')  

# Train the model on your dataset
model.train(
    data='config.yaml',  # Path to your dataset config file
    epochs=100,         # Number of epochs to train
    imgsz=640,         # Image size
    batch=8,           # Batch size
    workers=4          # Number of workers for data loading
)
