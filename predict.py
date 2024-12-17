from ultralytics import YOLO
import cv2

# Load the trained model
model_path = './runs/detect/train/weights/best.pt'  # Path to best model
model = YOLO(model_path)

# Path to the video
video_path = 'video.mp4'
output_path = 'output_video.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save the output video
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 model on each frame
    results = model(frame)

    # Visualize results
    annotated_frame = results[0].plot()  # Draw the detections on the frame
    out.write(annotated_frame)

    # Optional: Display the video in real-time
    cv2.imshow('Ant Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Detection complete. Saved to:", output_path)
