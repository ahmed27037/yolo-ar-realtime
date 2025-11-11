# YOLO AR Real-Time Detection

Real-time object detection and AR overlay system using YOLOv8. Supports multiple YOLO models with adjustable detection parameters.

## Overview

Computer vision project with real-time object detection and basic AR rendering:
- YOLOv8 object detection with multiple model sizes
- DeepSORT tracking
- Real-time AR overlays with transparency
- Adjustable confidence and IOU thresholds

## Features

- YOLOv8 object detection (nano to extra-large models)
- Multi-object tracking with DeepSORT
- AR overlay rendering with alpha blending
- Configurable detection thresholds
- Webcam and video file support

## Requirements

- Python 3.8+
- Webcam (for real-time demo)
- 4GB+ RAM

## Installation

Install dependencies and run. Works on Windows, Linux, and Mac.

### Windows

```powershell
cd path\to\cv_ar_graphics
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Windows CMD

```cmd
cd path\to\cv_ar_graphics
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Linux / Mac

```bash
cd /path/to/cv_ar_graphics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

YOLO models download automatically on first run.

## Usage

Activate venv first, then run:

```bash
# AR overlay (recommended)
python -m applications.ar_overlay --camera 0

# Object detection
python -m applications.object_detection --video 0

# Video file
python -m applications.object_detection --video input.mp4 --output output.mp4
```

Options:
```bash
# Use larger model
python -m applications.ar_overlay --camera 0 --model yolov8x

# Adjust thresholds
python -m applications.ar_overlay --camera 0 --confidence 0.5 --iou 0.4
```

Press `q` to quit.

## Examples

AR overlay on webcam:
```bash
python -m applications.ar_overlay --camera 0
```

Object detection on video:
```bash
python -m applications.object_detection --video input.mp4 --output output.mp4
```

With custom settings:
```bash
python -m applications.ar_overlay --camera 0 --model yolov8m --confidence 0.3
```

## Project Structure

```
cv_ar_graphics/
├── vision/                # Computer vision pipeline
│   ├── detection.py       # Object detection (YOLO)
│   ├── tracking.py        # Object tracking (DeepSORT)
│   └── segmentation.py    # Segmentation (SAM)
├── ar_renderer/           # AR rendering engine
│   ├── renderer.py        # Vulkan/OpenGL renderer
│   └── compositor.py      # Real-time compositing
├── applications/          # Demo applications
│   ├── object_detection.py
│   └── ar_overlay.py
└── requirements.txt       # Dependencies
```

## Usage Examples

### Python API

```python
from cv_ar_graphics.vision import YOLODetector, DeepSORTTracker
from cv_ar_graphics.ar_renderer import VulkanRenderer
import cv2

# Initialize components
detector = YOLODetector(model_type="yolov8n")
detector.load_model()
tracker = DeepSORTTracker()
renderer = VulkanRenderer(width=1920, height=1080)
renderer.initialize()

# Process frame
image = cv2.imread("image.jpg")
detections = detector.detect(image)
tracked = tracker.update(detections)

# Create AR overlays
overlays = []
for obj in tracked:
    overlays.append({
        'type': 'box',
        'bbox': obj['bbox'],
        'color': (0, 255, 0),
        'thickness': 2
    })

# Render
result = renderer.render_overlay(image, overlays)
cv2.imwrite("output.jpg", result)
```

### Custom AR Overlays

```python
from cv_ar_graphics.ar_renderer import Compositor
import numpy as np

compositor = Compositor(alpha=0.7)

# Create custom overlay
overlay = np.zeros((100, 100, 3), dtype=np.uint8)
overlay[:, :] = (0, 255, 0)  # Green overlay

# Compose
result = compositor.compose(background_image, overlay, mask=None)
```

## Available Models

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| yolov8n | 3MB | ~70ms | Fastest, good for testing |
| yolov8s | 11MB | ~100ms | Balanced |
| yolov8m | 26MB | ~200ms | Decent accuracy |
| yolov8l | 44MB | ~400ms | More accurate |
| yolov8x | 68MB | ~1200ms | Best accuracy |

Use `--model` flag to select:
```bash
python -m applications.ar_overlay --camera 0 --model yolov8x
```

## Configuration

Command-line options:
```bash
--model yolov8n|s|m|l|x   # Model size (default: yolov8n)
--confidence 0.25          # Detection threshold (default: 0.25)
--iou 0.45                 # IOU threshold (default: 0.45)
```

Examples:
```bash
# Higher confidence = fewer false positives
python -m applications.ar_overlay --camera 0 --confidence 0.6

# Lower confidence = detect more objects
python -m applications.ar_overlay --camera 0 --confidence 0.15
```

## Troubleshooting

**Camera not opening:**
Try different camera IDs (0, 1, 2) or check permissions.

**NumPy/OpenCV crashes:**
```bash
pip uninstall numpy -y
pip install "numpy>=2.0.0,<2.3.0" --force-reinstall
```

**Model download fails:**
Models download on first run. Manually download from ultralytics if needed.

## License

MIT

