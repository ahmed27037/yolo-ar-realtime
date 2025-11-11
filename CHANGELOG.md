# Changelog

## Recent Updates

### Features Added
- Camera support for object detection app
- Command-line options for model selection (yolov8n through yolov8x)
- Adjustable confidence and IOU thresholds
- AR overlay with transparency support

### Dependencies
- Streamlined requirements.txt to essential packages only
- Fixed NumPy compatibility (2.0.0-2.3.0 range)
- Optional development dependencies in requirements-dev.txt

### Documentation
- Simplified README
- Added usage examples
- Model comparison table

## Usage

AR overlay:
```bash
python -m applications.ar_overlay --camera 0
```

Object detection:
```bash
python -m applications.object_detection --video 0
```

With options:
```bash
python -m applications.ar_overlay --camera 0 --model yolov8x --confidence 0.5
```
