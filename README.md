# Object Detection using OpenCV and SSD MobileNet

This project demonstrates real-time object detection in images, videos, and webcam streams using a pre-trained SSD MobileNet model with OpenCV’s DNN module in Python.

## Table of Contents

- [Introduction](#introduction)
- [Project Objective](#project-objective)
- [Tools & Technologies Used](#tools--technologies-used)
- [Dataset & Model](#dataset--model)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

---

## Introduction

This project uses deep learning and computer vision techniques to detect and label objects in various media. It leverages OpenCV’s DNN module and a pre-trained SSD MobileNet model trained on the COCO dataset.

## Project Objective

- Detect and label objects in images, videos, and webcam streams.
- Draw bounding boxes and class names for each detected object.
- Demonstrate real-time object detection capabilities.

## Tools & Technologies Used

- Python 3.x
- OpenCV
- Matplotlib
- Pre-trained SSD MobileNet v3 model (TensorFlow)
- COCO dataset labels

## Dataset & Model

- **Model:** SSD MobileNet v3 (TensorFlow)
- **Config:** `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`
- **Weights:** `frozen_inference_graph.pb`
- **Labels:** `labels.txt` (COCO dataset class names)

## Project Structure

```
Project Object Detection/
│
├── detect.ipynb                # Main Jupyter notebook
├── frozen_inference_graph.pb   # Model weights
├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt  # Model config
├── labels.txt                  # Class labels
├── boy.jpg                     # Sample image
├── pixels geoge morina.mp4     # Sample video (optional)
└── README.md                   # Project documentation
```

## How to Run

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Project\ Object\ Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python matplotlib
   ```

3. **Download model files and labels:**
   - Place `frozen_inference_graph.pb`, `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`, and `labels.txt` in the project directory.

4. **Run the notebook:**
   - Open `detect.ipynb` in Jupyter Notebook or VS Code.
   - Run each cell sequentially to perform detection on images, videos, or webcam streams.

## Results

- The code displays detected objects in images and video frames with bounding boxes and class names.
- Real-time detection is supported via webcam.

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
