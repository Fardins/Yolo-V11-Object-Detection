# Cat & Dog Detection Using YOLOv11

<div style="display: flex; justify-content: space-between; align-items: center;">
  <!-- Left Side:Video -->
  <div style="flex: 1; margin-right: 10px;">
    <video width="100%" height="360" controls>
      <source src="./output/cat_output.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>

  <!-- Right Side:Video -->
  <div style="flex: 1; margin-left: 10px;">
    <video width="100%" height="360" controls>
      <source src="./output/dog_output.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>



## Overview
This project implements an object detection application using the YOLOv11 framework to identify and detect cats and dogs in video files. The pipeline consists of training the YOLOv11 model, processing video inputs for object detection, and displaying annotated output videos. Streamlit is used for a user-friendly web interface to upload videos and visualize results.

## Features
- **`Object Detection:`** Detects and identifies cats and dogs in video files.

- **`Web Interface:`** Easy-to-use interface for video uploads and results visualization.

- **`Video Processing:`** Handles various video formats (e.g., MP4, AVI) for input and output.

- **`Post-Processing:`** Uses FFmpeg for additional video processing steps.

- **`Custom Dataset:`** Trained on a dataset specifically configured for cat and dog detection.

## Project Structure

```bash
|-- streamlit_app/                 # Streamlit web app directory
    |-- app.py                     # Main Streamlit app for object detection
    |-- imageio_ffmpeg.py          # FFmpeg wrapper script for post-processing
    |-- predict.py                 # Script for running predictions on video files
    |-- train.py                   # Script for training the YOLOv11 model
    |-- config.yaml                # Dataset configuration file
    |-- requirements.txt           # Python dependencies

|-- runs/                          # Directory for YOLO training outputs
    |-- detect/train/weights/      # Trained YOLO model weights

|-- data/                          # Dataset directory
    |-- images/train               # Training images
    |-- images/val                 # Validation images

|-- input/                         # Input folder for prediction script
|-- output/                        # Output folder for annotated videos
|-- streamlit_output/              # Temporary folder for Streamlit outputs
```

## Installation

### Prerequisites

- Python 3.8 or higher

- Pip installed

### Step 1: Clone the Repository
```bash
git clone <repository_url>
cd <repository_directory>
```
### Step 2: Install Dependencies
Install required Python packages:
```bash
pip install -r requirements.txt
```
### Step 3: Prepare Dataset

- Add your training and validation data to the `data/images/train` and `data/images/val` directories, respectively.

- Update the `config.yaml` file with your dataset paths and class names.

## Usage
### Training the Model

Run the `train.py` script to train the YOLOv11 model:
```bash
python train.py
```
The trained model will be saved in the `runs/detect/train/weights/` directory.

### Running the Streamlit App
Start the Streamlit web app:
```bash
streamlit run app.py
```
- Upload a video file (MP4 or AVI).

- View the annotated video with detected objects in the Streamlit interface.

### Processing Videos with the Prediction Script

Use `predict.py` to process videos in batch mode:
```bash
python predict.py
```
- Place your input videos in the `input/` directory.

- The processed videos will be saved in the `output/` directory.

## Dataset Configuration

The `config.yaml` file defines the dataset and class labels:
```bash
path: E:\Deep Learning & Generative AI\Projects\Yolo-V11\Yolo-V11-Object-Detection\data
train: images/train
val: images/val
nc: 2
names: ['cat', 'dog']
```
- `path:` Base directory of the dataset.

- `train:` Path to training images.

- `val:` Path to validation images.

- `nc:` Number of classes.

- `names:` Names of the classes.

## Output Example

After processing a video, the output will be an annotated video file showing detected objects with bounding boxes and labels. The output video is displayed in the Streamlit app or saved in the `output/` folder for batch processing.

You can view these examples within the Streamlit app or try it yourself by selecting an image from the dropdown and running segmentation.

## Dependencies

- Ultralytics YOLO
- Streamlit
- OpenCV
- ImageIO with FFmpeg
- Install these dependencies via `requirements.txt`:
```bash
pip install -r requirements.txt
```
## Acknowledgments

## Acknowledgments
- **YOLO Framework:** [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) serves as the base framework.
- **FFmpeg:** For video post-processing.


## Future Improvements

- Enhance the dataset for better accuracy.

- Add support for more object classes.

- Optimize the Streamlit app for real-time performance.

- Extend the project for live camera feeds.
