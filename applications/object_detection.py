import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
import logging

from vision import YOLODetector, DeepSORTTracker
from ar_renderer import VulkanRenderer, Compositor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetectionDemo:
    def __init__(self, model_type: str = "yolov8n", confidence: float = 0.25, iou: float = 0.45):
        self.detector = YOLODetector(model_type=model_type)
        self.detector.confidence_threshold = confidence
        self.detector.iou_threshold = iou
        self.tracker = DeepSORTTracker()
        self.renderer = None
        self.compositor = Compositor()
    
    def run_video(self, video_path: str, output_path: Optional[str] = None, display: bool = True):
        # Try to convert to int for camera index, otherwise use as file path
        try:
            video_source = int(video_path)
        except ValueError:
            video_source = video_path
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize renderer
        self.renderer = VulkanRenderer(width, height)
        self.renderer.initialize()
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                detections = self.detector.detect(frame)
                tracked = self.tracker.update(detections)
                
                # Render
                overlays = []
                for obj in tracked:
                    bbox = obj['bbox']
                    class_name = obj.get('class_name', 'object')
                    track_id = obj.get('track_id', -1)
                    confidence = obj.get('confidence', 0.0)
                    
                    overlays.append({
                        'type': 'box',
                        'bbox': bbox,
                        'color': (0, 255, 0),
                        'thickness': 2,
                    })
                    
                    overlays.append({
                        'type': 'text',
                        'text': f"{class_name} #{track_id} ({confidence:.2f})",
                        'position': [bbox[0], bbox[1] - 10],
                        'color': (0, 255, 0),
                    })
                
                result = self.renderer.render_overlay(frame, overlays)
                
                if display:
                    cv2.imshow('Object Detection Demo', result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if writer:
                    writer.write(result)
                
                if frame_count % 30 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            logger.info(f"Processed {frame_count} frames total")


def main():
    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument("--video", type=str, required=True, 
                       help="Input video path or camera index (0, 1, 2, etc.)")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--model", type=str, default="yolov8n", 
                       help="YOLO model type (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser.add_argument("--confidence", type=float, default=0.25, 
                       help="Confidence threshold (0.0-1.0, default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, 
                       help="IOU threshold for NMS (0.0-1.0, default: 0.45)")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    
    args = parser.parse_args()
    
    demo = ObjectDetectionDemo(model_type=args.model, confidence=args.confidence, iou=args.iou)
    demo.run_video(args.video, args.output, display=not args.no_display)


if __name__ == "__main__":
    main()

