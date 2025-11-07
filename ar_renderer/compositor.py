"""
AR Compositor

Provides real-time compositing of AR overlays with alpha blending.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Compositor:
    """
    Real-time AR compositor
    
    Handles alpha blending and overlay composition
    """
    
    def __init__(self, alpha: float = 0.7):
        """
        Initialize compositor
        
        Args:
            alpha: Default alpha blending factor (0.0 to 1.0)
        """
        self.alpha = alpha
    
    def compose(self, background: np.ndarray, overlay: np.ndarray,
               mask: Optional[np.ndarray] = None, alpha: Optional[float] = None) -> np.ndarray:
        """
        Compose overlay on background with alpha blending
        
        Args:
            background: Background image
            overlay: Overlay image
            mask: Optional mask for selective blending
            alpha: Alpha blending factor (overrides default)
        
        Returns:
            Composed image
        """
        if alpha is None:
            alpha = self.alpha
        
        # Ensure same size
        if overlay.shape[:2] != background.shape[:2]:
            overlay = cv2.resize(overlay, (background.shape[1], background.shape[0]))
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape[:2] != background.shape[:2]:
                mask = cv2.resize(mask, (background.shape[1], background.shape[0]))
            if len(mask.shape) == 2:
                mask = mask[:, :, np.newaxis]
            overlay = overlay * mask
        
        # Alpha blending
        result = cv2.addWeighted(background, 1 - alpha, overlay, alpha, 0)
        
        return result
    
    def compose_overlays(self, background: np.ndarray, overlays: List[Dict]) -> np.ndarray:
        """
        Compose multiple overlays on background
        
        Args:
            background: Background image
            overlays: List of overlay dictionaries with 'image', 'position', 'alpha', etc.
        
        Returns:
            Composed image
        """
        result = background.copy()
        
        for overlay in overlays:
            overlay_image = overlay.get('image')
            if overlay_image is None:
                continue
            
            position = overlay.get('position', (0, 0))
            alpha = overlay.get('alpha', self.alpha)
            mask = overlay.get('mask')
            
            # Place overlay at position
            x, y = position
            h, w = overlay_image.shape[:2]
            
            if x + w <= result.shape[1] and y + h <= result.shape[0]:
                roi = result[y:y+h, x:x+w]
                composed = self.compose(roi, overlay_image, mask, alpha)
                result[y:y+h, x:x+w] = composed
        
        return result

