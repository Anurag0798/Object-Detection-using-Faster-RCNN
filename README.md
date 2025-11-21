# Object Detection Using Faster R-CNN

A practical, real-time object detection project using the state-of-the-art Faster R-CNN (Region-Based Convolutional Neural Network) framework via PyTorch and TorchVision. This demo showcases how to perform bounding box detection and label visualization for common objects in live video streams (such as from a webcam), ideal for learning, prototyping, and research in powerful deep learning–based computer vision.

## Features

- **Real-Time Object Detection:**
  Utilizes the `fasterrcnn_resnet50_fpn` model from TorchVision with pretrained weights for instant detection of multiple object types, including people, vehicles, animals, and common everyday objects.
- **Seamless Webcam Integration:**
  Captures frames live from a connected webcam and overlays detected bounding boxes and class labels.
- **Confidence Thresholding:**
  Filters out low-confidence predictions (default threshold = 0.7) for robust results.
- **COCO-Class Mapping:**
  Supports mapping of output indices to readable COCO object class names.
- **Minimal, Easy-to-Adapt Python Script:**
  Clear, concise implementation for quick customization and experimentation.

## Repository Structure

```
.
├── Object_Detection_using_Faster_RCNN.py   # Main script for running webcam-based detection
└── README.md                               # Project documentation (this file)
```

## Requirements

- Python 3.7+
- torch          (PyTorch)
- torchvision
- opencv-python

Install dependencies with:
    ```bash
    pip install torch torchvision opencv-python
    ```

## Getting Started

1. **Clone the repository**
    ```bash
    git clone https://github.com/Anurag0798/Object-Detection-using-Faster-RCNN.git

    cd Object-Detection-using-Faster-RCNN
    ```

2. **(Optional) Create and activate a virtual environment**
    ```bash
    python -m venv .venv

    # On macOS/Linux
    source .venv/bin/activate

    # On Windows
    .venv\Scripts\Activate.ps1
    ```

3. **Install requirements**
    ```bash
    pip install torch torchvision opencv-python
    ```

4. **Connect a webcam** (if running live detection).

5. **Run the main detection script**
    ```bash
    python Object_Detection_using_Faster_RCNN.py
    ```

    - The real-time webcam stream will open in a window, showing detection results with bounding boxes and class labels.
    - Press the `'q'` key to quit.

## How It Works

- Loads a pretrained Faster R-CNN model (`fasterrcnn_resnet50_fpn`) from TorchVision.
- Captures video frames using OpenCV.
- Converts each frame to a tensor and runs detection.
- Draws bounding boxes with confidence > 0.7 and their corresponding class names (using COCO Instance category mapping).
- Displays annotated frames in a window in real time.

> **Note:** If using a different camera, change the index in `cv2.VideoCapture(1)` to `cv2.VideoCapture(0)` or appropriate.

## Customization

- **Adjust Confidence Threshold:**
  Change the `score > 0.7` line to raise/lower detection strictness.
- **Different Video Source:**
  Change the `cv2.VideoCapture` index for another camera or to a file path for processing a saved video.
- **Save Detections:**
  Add logic in the main loop to save images or detection results as needed.
- **Class Mapping:**
  Update `coco_instance_catagory_names` if you want to use custom class lists.

## License

This code is for educational and research purposes.  
Add or update a LICENSE file if you wish to specify usage rights.

## Acknowledgements

- PyTorch and TorchVision teams for model implementations and pretrained weights
- OpenCV for real-time computer vision utilities

## Contributing

Contributions and suggestions are welcome!  
Feel free to fork, modify, and open a pull request.  
If you found this project useful, please star the repo.