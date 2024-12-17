import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile

st.title("Ant Detection Using YOLOv11")
st.write("Upload a video to detect ants.")

# Load the trained model
@st.cache_resource
def load_model():
    model = YOLO('./runs/detect/train/weights/best.pt')
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    # Open the video and perform detection
    st.write("Processing video...")
    cap = cv2.VideoCapture(video_path)
    output_path = "output_video.mp4"
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    # Display output video
    st.video(output_path)
    st.success("Detection complete!")
