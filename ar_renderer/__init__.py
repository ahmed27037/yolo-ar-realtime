"""
AR Rendering Engine

Provides Vulkan-based AR rendering with real-time compositing.
"""

from .renderer import ARRenderer, VulkanRenderer
from .compositor import Compositor

__all__ = [
    'ARRenderer',
    'VulkanRenderer',
    'Compositor',
]

