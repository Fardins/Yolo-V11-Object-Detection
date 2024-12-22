import streamlit as st
from ultralytics import YOLO
import cv2
import os
import tempfile
import subprocess
from datetime import datetime

st.title("Cat & Dog Detection Using YOLOv11")
st.write("Upload a video to detect objects:")

@st.cache_resource
def load_model():
    model = YOLO('./runs/detect/train/weights/best.pt')
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi"])

if uploaded_file:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    cap = cv2.VideoCapture(video_path)

    # Generate a unique name for the output file using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    yolo_output_path = f"./streamlit_output/yolo_output_{timestamp}.mp4"
    ffmpeg_output_path = f"./streamlit_output/ffmpeg_output_{timestamp}.mp4"

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer
    out = cv2.VideoWriter(yolo_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    st.write("Post-processing video...")
    # Call the second script and pass the YOLO output video path
    subprocess.run(['python', 'imageio_ffmpeg.py', yolo_output_path, ffmpeg_output_path])

    # Check if the file exists
    if os.path.exists(ffmpeg_output_path):
        st.video(ffmpeg_output_path)
        st.success("Object detection complete! Here's a preview of your processed video.")
    else:
        st.error("Processed video file not found. Please check the file path.")
