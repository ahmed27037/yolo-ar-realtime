"""
Object Detection

Provides real-time object detection using YOLO and other models.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Base object detector class"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize object detector
        
        Args:
            model_path: Path to model file
        """
        self.model = None
        self.model_path = model_path
        self.classes = []
    
    def load_model(self, model_path: str) -> None:
        """Load detection model"""
        raise NotImplementedError
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detection dictionaries with 'bbox', 'confidence', 'class_id', 'class_name'
        """
        raise NotImplementedError


class YOLODetector(ObjectDetector):
    """
    YOLO-based object detector
    
    Supports YOLOv8 and other YOLO variants
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "yolov8n"):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file
            model_type: YOLO model type ("yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x")
        """
        super().__init__(model_path)
        self.model_type = model_type
        self.model = None
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            
            if model_path is None:
                # Load pre-trained model
                model_path = f"{self.model_type}.pt"
            
            self.model = YOLO(model_path)
            logger.info(f"Loaded YOLO model: {model_path}")
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.classes = list(self.model.names.values())
            
        except ImportError:
            logger.warning("ultralytics not available, using OpenCV DNN fallback")
            self._load_opencv_model(model_path)
    
    def _load_opencv_model(self, model_path: Optional[str]) -> None:
        """Load YOLO model using OpenCV DNN"""
        try:
            # This is a fallback for when ultralytics is not available
            # In production, would load YOLO weights and config
            logger.info("Using OpenCV DNN backend (fallback)")
            self.model = "opencv_dnn"  # Placeholder
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects using YOLO
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detections
        """
        if self.model is None:
            self.load_model()
        
        detections = []
        
        try:
            if isinstance(self.model, str) and self.model == "opencv_dnn":
                # OpenCV DNN fallback
                detections = self._detect_opencv(image)
            else:
                # Use ultralytics YOLO
                results = self.model(image, conf=self.confidence_threshold, iou=self.iou_threshold)
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name,
                        })
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        
        return detections
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict]:
        """Fallback detection using OpenCV"""
        # Simplified detection using OpenCV
        # In production, would use full YOLO DNN implementation
        detections = []
        
        # Placeholder: would implement actual YOLO DNN inference
        logger.debug("Using OpenCV DNN fallback detection")
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection bounding boxes on image
        
        Args:
            image: Input image
            detections: List of detections
        
        Returns:
            Image with drawn bounding boxes
        """
        result = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            confidence = det['confidence']
            class_name = det.get('class_name', 'object')
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result

