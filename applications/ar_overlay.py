"""
AR Overlay Demo Application
"""

import cv2
import argparse
import numpy as np
from typing import Optional
import logging

from vision import YOLODetector, DeepSORTTracker
from ar_renderer import VulkanRenderer, Compositor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AROverlayDemo:
    """AR overlay demo application"""
    
    def __init__(self, model_type: str = "yolov8n", confidence: float = 0.25, iou: float = 0.45):
        """
        Initialize demo
        
        Args:
            model_type: YOLO model type
            confidence: Confidence threshold for detections
            iou: IOU threshold for non-maximum suppression
        """
        self.detector = YOLODetector(model_type=model_type)
        self.detector.confidence_threshold = confidence
        self.detector.iou_threshold = iou
        self.tracker = DeepSORTTracker()
        self.renderer = None
        self.compositor = Compositor(alpha=0.6)
    
    def run_camera(self, camera_id: int = 0, display: bool = True):
        """
        Run AR overlay on camera feed
        
        Args:
            camera_id: Camera device ID
            display: Whether to display results
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera: {camera_id}")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize renderer
        self.renderer = VulkanRenderer(width, height)
        self.renderer.initialize()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect objects
                detections = self.detector.detect(frame)
                
                # Track objects
                tracked = self.tracker.update(detections)
                
                # Create AR overlays
                image_overlays = []
                text_overlays = []
                
                for obj in tracked:
                    bbox = obj['bbox']
                    class_name = obj.get('class_name', 'object')
                    track_id = obj.get('track_id', -1)
                    
                    # Create overlay with semi-transparent box
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    
                    # Ensure valid bbox dimensions
                    if x2 > x1 and y2 > y1:
                        overlay = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
                        overlay[:, :] = (0, 255, 0)  # Green overlay
                        
                        image_overlays.append({
                            'image': overlay,
                            'position': (x1, y1),
                            'alpha': 0.3,
                        })
                    
                    # Store text label for later rendering
                    text_overlays.append({
                        'text': f"{class_name} #{track_id}",
                        'position': (x1, max(10, y1 - 10)),
                        'color': (0, 255, 0),
                    })
                
                # Compose image overlays
                result = self.compositor.compose_overlays(frame, image_overlays)
                
                # Draw text overlays on top
                for text_overlay in text_overlays:
                    cv2.putText(result, text_overlay['text'], 
                               text_overlay['position'],
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.6, text_overlay['color'], 2)
                
                # Display
                if display:
                    cv2.imshow('AR Overlay Demo', result)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AR Overlay Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--model", type=str, default="yolov8n", 
                       help="YOLO model type (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)")
    parser.add_argument("--confidence", type=float, default=0.25, 
                       help="Confidence threshold (0.0-1.0, default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, 
                       help="IOU threshold for NMS (0.0-1.0, default: 0.45)")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    
    args = parser.parse_args()
    
    demo = AROverlayDemo(model_type=args.model, confidence=args.confidence, iou=args.iou)
    demo.run_camera(args.camera, display=not args.no_display)


if __name__ == "__main__":
    main()

