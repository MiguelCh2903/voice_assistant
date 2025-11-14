"""
Type definitions for the voice assistant core system.

This module defines data structures used throughout the voice assistant system.
These types serve as internal representations and can be converted to/from
voice_assistant_msgs ROS2 message types for cross-package communication.

Design principles:
- Keep types simple and focused on single responsibilities
- Use dataclasses for immutable data structures where possible
- Include comprehensive docstrings for ROS2 integration clarity
- Maintain compatibility with voice_assistant_msgs message definitions

Usage:
    These types are used internally by the voice assistant core system and
    can be converted to ROS2 messages using the ros_utils module.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum, auto
import time


class AssistantState(Enum):
    """States of the voice assistant finite state machine."""

    DISCONNECTED = auto()  # No connection to ESPHome device
    CONNECTING = auto()  # Attempting to connect to device
    IDLE = auto()  # Connected and waiting for wake word
    STREAMING_AUDIO = auto()  # Receiving audio stream from device
    TRANSCRIBING = auto()  # Processing audio for speech-to-text
    PROCESSING_LLM = auto()  # Processing transcription with LLM
    PLAYING_RESPONSE = auto()  # Playing TTS response back to device
    ERROR = auto()  # Error state requiring recovery


class VoiceEventType(Enum):
    """Types of voice assistant events."""

    WAKE_WORD_DETECTED = auto()
    AUDIO_STREAM_START = auto()
    AUDIO_STREAM_END = auto()
    TRANSCRIPTION_COMPLETE = auto()
    LLM_RESPONSE_READY = auto()
    TTS_PLAYBACK_START = auto()
    TTS_PLAYBACK_END = auto()
    CONNECTION_ESTABLISHED = auto()
    CONNECTION_LOST = auto()
    ERROR_OCCURRED = auto()
    HEARTBEAT = auto()


class AudioFormat(Enum):
    """Supported audio formats."""

    PCM_16KHZ_16BIT_MONO = auto()
    WAV = auto()


@dataclass
class AudioChunk:
    """
    Raw audio data chunk from ESPHome device with ROS2 integration support.

    This class represents a single chunk of audio data in the voice processing
    pipeline. It is compatible with voice_assistant_msgs/AudioChunk.msg and
    includes validation and utility methods for ROS2 integration.

    Attributes:
        data: Raw audio bytes from ESPHome device
        timestamp: When chunk was received (Unix timestamp in seconds)
        sequence_id: Sequential chunk identifier within a stream
        format: Audio format type (enum for consistency)
        sample_rate: Audio sample rate in Hz (typically 16000)
        channels: Number of audio channels (typically 1 for mono)
        sample_width: Bits per sample (typically 16 for 16-bit PCM)
        is_final: True if this is the last chunk in the audio stream

    Note:
        This type is designed to be easily converted to/from ROS2 messages
        using the ros_utils module functions.
    """

    data: bytes
    timestamp: float
    sequence_id: int
    format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT_MONO
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 16
    is_final: bool = False

    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.timestamp <= 0:
            self.timestamp = time.time()

        # Validate audio parameters for ROS2 compatibility
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample_rate: {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"Invalid channels: {self.channels}")
        if self.sample_width <= 0:
            raise ValueError(f"Invalid sample_width: {self.sample_width}")
        if not isinstance(self.data, bytes):
            raise TypeError(f"data must be bytes, got {type(self.data)}")

    @property
    def size_bytes(self) -> int:
        """Get the size of audio data in bytes."""
        return len(self.data)

    @property
    def duration_seconds(self) -> float:
        """Calculate audio duration in seconds based on format."""
        if len(self.data) == 0:
            return 0.0

        bytes_per_sample = self.sample_width // 8
        bytes_per_frame = bytes_per_sample * self.channels
        frames = len(self.data) // bytes_per_frame
        return frames / self.sample_rate

    def validate_format_consistency(self) -> bool:
        """
        Validate that chunk parameters match the declared audio format.

        Returns:
            True if format is consistent, False otherwise
        """
        if self.format == AudioFormat.PCM_16KHZ_16BIT_MONO:
            return (
                self.sample_rate == 16000
                and self.sample_width == 16
                and self.channels == 1
            )
        return True  # Other formats not strictly validated yet


@dataclass
class TranscriptionResult:
    """
    Result from speech-to-text processing.

    Attributes:
        text: Transcribed text from audio
        confidence: Confidence score (0.0 to 1.0)
        language: Detected language code
        processing_time: Time taken for transcription
        error_message: Error description if transcription failed
        audio_duration: Duration of audio that was transcribed
    """

    text: str
    confidence: float = 0.0
    language: str = "en"
    processing_time: float = 0.0
    error_message: Optional[str] = None
    audio_duration: float = 0.0


@dataclass
class LLMResponse:
    """
    Response from large language model processing.

    Attributes:
        response_text: Generated response text
        intent: Detected intent/action
        entities: Extracted entities from user input
        conversation_id: Unique conversation identifier
        continue_conversation: Whether to continue listening
        processing_time: Time taken for LLM processing
        confidence: Confidence in response
        error_message: Error description if processing failed
    """

    response_text: str
    intent: str = ""
    entities: Dict[str, Any] = None
    conversation_id: str = ""
    continue_conversation: bool = False
    processing_time: float = 0.0
    confidence: float = 0.0
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = {}


@dataclass
class VoiceEvent:
    """
    General voice assistant system event.

    Attributes:
        event_type: Type of event that occurred
        timestamp: When event occurred
        data: Additional event data
        source: Component that generated the event
        message: Human-readable event description
    """

    event_type: VoiceEventType
    timestamp: float
    data: Dict[str, Any] = None
    source: str = "voice_assistant_core"
    message: str = ""

    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class ESPHomeDeviceInfo:
    """
    Information about connected ESPHome device.

    Attributes:
        host: Device IP address or hostname
        port: Device port number
        password: API password if required
        encryption_key: API encryption key if required
        device_name: Human-readable device name
        mac_address: Device MAC address
        esphome_version: ESPHome firmware version
        supported_features: List of supported voice assistant features
        is_connected: Current connection status
    """

    host: str
    port: int = 6053
    password: Optional[str] = None
    encryption_key: Optional[str] = None
    device_name: str = "ESPHome Voice Assistant"
    mac_address: str = ""
    esphome_version: str = ""
    supported_features: List[str] = None
    is_connected: bool = False

    def __post_init__(self):
        if self.supported_features is None:
            self.supported_features = []


@dataclass
class ConnectionState:
    """
    State of connection to ESPHome device.

    Attributes:
        is_connected: Whether currently connected
        connection_attempts: Number of connection attempts made
        last_connection_time: Timestamp of last successful connection
        last_error: Last connection error message
        reconnect_delay: Current reconnection delay in seconds
        max_reconnect_delay: Maximum reconnection delay
    """

    is_connected: bool = False
    connection_attempts: int = 0
    last_connection_time: float = 0.0
    last_error: Optional[str] = None
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0


@dataclass
class SystemConfig:
    """
    Voice assistant system configuration.

    Attributes:
        device_info: ESPHome device connection information
        audio_buffer_size: Maximum audio buffer size in bytes
        connection_timeout: Connection timeout in seconds
        heartbeat_interval: Heartbeat check interval in seconds
        max_audio_duration: Maximum audio stream duration in seconds
        transcription_timeout: Timeout for transcription processing
        llm_timeout: Timeout for LLM processing
        enable_debug_logging: Whether to enable debug logging
    """

    device_info: ESPHomeDeviceInfo
    audio_buffer_size: int = 1024 * 1024  # 1MB
    connection_timeout: float = 10.0
    heartbeat_interval: float = 30.0
    max_audio_duration: float = 30.0
    transcription_timeout: float = 15.0
    llm_timeout: float = 10.0
    enable_debug_logging: bool = False
