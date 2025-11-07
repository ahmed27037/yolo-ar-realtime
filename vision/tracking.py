"""
Object Tracking

Provides multi-object tracking using DeepSORT and other algorithms.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ObjectTracker:
    """Base object tracker class"""
    
    def __init__(self):
        """Initialize tracker"""
        self.tracks = {}
        self.next_id = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections from detector
        
        Returns:
            List of tracked objects with track IDs
        """
        raise NotImplementedError


class DeepSORTTracker(ObjectTracker):
    """
    DeepSORT-based multi-object tracker
    
    Combines Kalman filtering with deep appearance features
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3):
        """
        Initialize DeepSORT tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is confirmed
        """
        super().__init__()
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections
        
        Returns:
            List of tracked objects
        """
        self.frame_count += 1
        
        # Simplified DeepSORT implementation
        # Full implementation would use Kalman filter and appearance features
        
        tracked_objects = []
        
        # Match detections to existing tracks
        matched_tracks = set()
        for det in detections:
            bbox = det['bbox']
            center = self._bbox_center(bbox)
            
            # Find best matching track
            best_track_id = None
            best_distance = float('inf')
            
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                # Calculate IoU or distance
                distance = self._calculate_distance(center, track['predicted_center'])
                
                if distance < best_distance and distance < 100:  # Threshold
                    best_distance = distance
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                track = self.tracks[best_track_id]
                track['bbox'] = bbox
                track['predicted_center'] = center
                track['age'] = 0
                track['hits'] += 1
                
                det['track_id'] = best_track_id
                det['age'] = track['age']
                det['hits'] = track['hits']
                tracked_objects.append(det)
                matched_tracks.add(best_track_id)
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'bbox': bbox,
                    'predicted_center': center,
                    'age': 0,
                    'hits': 1,
                }
                
                det['track_id'] = track_id
                det['age'] = 0
                det['hits'] = 1
                tracked_objects.append(det)
                matched_tracks.add(track_id)
        
        # Update unmatched tracks
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track['age'] += 1
                if track['age'] > self.max_age:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Filter by min_hits
        tracked_objects = [obj for obj in tracked_objects if obj['hits'] >= self.min_hits]
        
        return tracked_objects
    
    def _bbox_center(self, bbox: List[int]) -> tuple:
        """Calculate bounding box center"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _calculate_distance(self, p1: tuple, p2: tuple) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


class SimpleTracker(ObjectTracker):
    """
    Simple IoU-based tracker (fallback)
    """
    
    def __init__(self, iou_threshold: float = 0.3):
        """
        Initialize simple tracker
        
        Args:
            iou_threshold: IoU threshold for matching
        """
        super().__init__()
        self.iou_threshold = iou_threshold
    
    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracker using IoU matching"""
        tracked_objects = []
        
        for det in detections:
            bbox = det['bbox']
            
            # Find best matching track
            best_track_id = None
            best_iou = 0
            
            for track_id, track in self.tracks.items():
                iou = self._calculate_iou(bbox, track['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update track
                self.tracks[best_track_id]['bbox'] = bbox
                det['track_id'] = best_track_id
            else:
                # New track
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {'bbox': bbox}
                det['track_id'] = track_id
            
            tracked_objects.append(det)
        
        return tracked_objects
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

