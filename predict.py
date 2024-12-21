from ultralytics import YOLO
import cv2
import os

# Load the trained model
model_path = './runs/detect/train/weights/best.pt'  # Path to best model
model = YOLO(model_path)

# Input and output folder paths
input_folder = './input/'
output_folder = './output/'
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# Process each video in the input folder
for video_file in os.listdir(input_folder):
    if not video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Only process video files
        continue

    video_path = os.path.join(input_folder, video_file)
    output_path = os.path.join(output_folder, os.path.splitext(video_file)[0] + '_output.mp4')

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

        # Run YOLO model on each frame
        results = model(frame)

        # Visualize results
        annotated_frame = results[0].plot()  # Draw the detections on the frame
        out.write(annotated_frame)

        # Optional: Display the video in real-time
        cv2.imshow('Video Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    print(f"Processed and saved: {output_path}")

cv2.destroyAllWindows()
print("All videos processed.")
