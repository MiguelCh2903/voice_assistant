"""
Voice Assistant Core package for ROS2.

VAPI-based voice assistant integration with ESPHome.
"""

from .vapi_voice_assistant_node import VapiVoiceAssistantNode, main
from .communication import ESPHomeClientWrapper
from .audio.types import (
    AudioChunk,
    ESPHomeDeviceInfo,
    AudioFormat,
)
from .vapi import VapiClient

__version__ = "0.1.0"
__author__ = "Voice Assistant Core Team"

__all__ = [
    "VapiVoiceAssistantNode",
    "main",
    "ESPHomeClientWrapper",
    "VapiClient",
    "AudioChunk",
    "ESPHomeDeviceInfo",
    "AudioFormat",
]
