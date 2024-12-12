# YOLO Object Detection Program

This program uses the YOLO (You Only Look Once) model for real-time object detection and detection on static images. It is implemented using OpenCV's DNN module and works with pre-trained YOLO weights and configuration files.

## Features:
- **Real-time Object Detection**: Uses your webcam to detect objects in real-time.
- **Static Image Object Detection**: Detects objects in a given image.
- **Bounding Boxes and Labels**: Draws bounding boxes around detected objects with their class labels.

## Requirements:
- Python 3.x
- OpenCV (`cv2` library)
- YOLOv3 model files:
  - `yolov3.weights`: Pre-trained weights for the YOLOv3 model.
  - `yolov3.cfg`: Configuration file for YOLOv3.
  - `coco.names`: File containing class labels for the COCO dataset.

## Installation:

1. Install Python 3.x from [here](https://www.python.org/downloads/).
2. Install OpenCV by running:
    ```bash
    pip install opencv-python
    ```
3. Download the required YOLOv3 files:
    - **Weights**: [Download yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
    - **Config File**: [Download yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
    - **Class Labels**: [Download coco.names](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)

## Usage:

### 1. Static Image Object Detection:
   - Choose the first option (`1`) when prompted.
   - Enter the path to your image file.
   - The program will detect and label objects in the image, drawing bounding boxes around them.

### 2. Real-time Webcam Object Detection:
   - Choose the second option (`2`) when prompted.
   - The program will use your webcam to detect objects in real-time and display them on the screen.
   - Press `q` to stop the webcam feed.

## Code Explanation:
- **`load_yolo_model()`**: Loads the YOLOv3 model with pre-trained weights and configuration files.
- **`detect_objects()`**: Processes the image and detects objects using the YOLO model, returning their bounding box coordinates, confidence, and class IDs.
- **`draw_labels()`**: Draws bounding boxes and labels on the image for the detected objects.
- **`object_detection()`**: Detects objects in a static image and displays the result.
- **`real_time_object_detection()`**: Performs object detection in real-time using the webcam.

## Example:

For static image detection:
```bash
Choose detection mode:
1. Static Image Detection
2. Real-time Webcam Detection
Enter choice (1 or 2): 1
Enter the image file path: path/to/your/image.jpg
