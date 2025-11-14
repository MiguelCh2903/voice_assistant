"""
Voice Assistant Core package for ROS2.

This package provides the core brain functionality for a distributed voice
assistant system built on ROS2 with ESPHome integration.

Main Components:
- VoiceAssistantNode: Main ROS2 node coordinating all components
- VoiceAssistantStateMachine: Finite state machine for conversation flow
- ESPHomeClientWrapper: Robust ESPHome device communication
- AudioBuffer: Audio stream buffering and management

Key Features:
- Robust ESPHome device connection with auto-reconnection
- Finite state machine for conversation state management
- Audio streaming and buffering from ESPHome devices
- ROS2 integration for distributed voice assistant system
- Error handling and recovery mechanisms
"""

from .voice_assistant_node import VoiceAssistantNode, main
from .state import VoiceAssistantStateMachine
from .communication import ESPHomeClientWrapper
from .audio import AudioBuffer
from .audio.types import (
    AudioChunk,
    AssistantState,
    TranscriptionResult,
    LLMResponse,
    VoiceEvent,
    VoiceEventType,
    ESPHomeDeviceInfo,
    SystemConfig,
    AudioFormat,
)

__version__ = "0.1.0"
__author__ = "Voice Assistant Core Team"
__email__ = "support@voice-assistant.dev"

__all__ = [
    # Main node
    "VoiceAssistantNode",
    "main",
    # Core components
    "VoiceAssistantStateMachine",
    "ESPHomeClientWrapper",
    "AudioBuffer",
    # Types and enums
    "AssistantState",
    "AudioChunk",
    "TranscriptionResult",
    "LLMResponse",
    "VoiceEvent",
    "VoiceEventType",
    "ESPHomeDeviceInfo",
    "SystemConfig",
    "AudioFormat",
    # ROS2 utilities
    "ros_utils",
    "json_utils",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]
