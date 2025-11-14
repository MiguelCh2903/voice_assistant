"""
System constants and configuration defaults for voice assistant core.

This module contains all configurable constants used throughout the voice
assistant system, including timeouts, audio settings, and connection parameters.
"""

# Connection and Network Settings
DEFAULT_ESPHOME_PORT = 6053
CONNECTION_TIMEOUT_SEC = 10.0
CONFIG_TIMEOUT_SEC = 5.0
HEARTBEAT_INTERVAL_SEC = 30.0
KEEPALIVE_INTERVAL_SEC = 15.0

# Reconnection Settings
INITIAL_RECONNECT_DELAY_SEC = 1.0
MAX_RECONNECT_DELAY_SEC = 60.0
RECONNECT_BACKOFF_MULTIPLIER = 1.5
MAX_RECONNECTION_ATTEMPTS = 0  # 0 = infinite attempts

# Audio Processing Settings - Optimized for Raspberry Pi 5
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SAMPLE_WIDTH = 16  # bits
DEFAULT_CHANNELS = 1
AUDIO_CHUNK_SIZE = 2048  # Optimal chunk size for streaming
MAX_AUDIO_BUFFER_SIZE = 1024 * 1024  # 1MB circular buffer
MAX_AUDIO_DURATION_SEC = 30.0  # Maximum audio buffer duration
AUDIO_QUEUE_MAX_SIZE = 50  # Deque size for circular buffer

# Voice Activity Detection (VAD) Settings
VAD_FRAME_LENGTH = 512  # Cobra VAD requires 512 samples per frame
VAD_THRESHOLD = 0.6  # Voice probability threshold (0.0 - 1.0)

# Turn Detection Settings - Two-phase intelligent detection
# Phase 1: Wait for consecutive speech burst to trigger turn start
# Phase 2: Analyze during silence with ML model + fallback timeout
TURN_DETECTOR_ENABLED = True
TURN_DETECTOR_SPEECH_START_DURATION = (
    0.7  # Consecutive speech to trigger turn start (seconds)
)
TURN_DETECTOR_SILENCE_ANALYSIS_DURATION = 1.5  # Silence before ML analysis (seconds)
TURN_DETECTOR_SILENCE_FALLBACK_TIMEOUT = (
    3.0  # Max silence before forced turn end (seconds)
)
TURN_DETECTOR_CONTEXT_BUFFER_DURATION = 8.0  # Audio context buffer duration (seconds)
TURN_DETECTOR_PRE_BUFFER_DURATION = 0.2  # Pre-buffer before speech detection (seconds)
TURN_DETECTOR_CONFIDENCE_THRESHOLD = 0.82  # ML confidence threshold (0-1)
TURN_DETECTOR_MIN_TURN_COOLDOWN = 2.5  # Minimum time between turn completions (seconds)
TURN_DETECTOR_ENABLE_WARMUP = True  # Run model warmup on initialization
TURN_DETECTOR_LOG_LEVEL = "INFO"  # Logging level (INFO/WARNING)

# Audio Format Settings
SUPPORTED_AUDIO_FORMATS = {
    "pcm": {"sample_rate": 16000, "sample_width": 16, "channels": 1},
    "wav": {"sample_rate": 16000, "sample_width": 16, "channels": 1},
}

# Processing Timeouts
STT_PROCESSING_TIMEOUT_SEC = 15.0
LLM_PROCESSING_TIMEOUT_SEC = 10.0
TTS_PROCESSING_TIMEOUT_SEC = 20.0
PIPELINE_TOTAL_TIMEOUT_SEC = 60.0

# State Machine Settings
STATE_TRANSITION_TIMEOUT_SEC = 5.0
ERROR_RECOVERY_DELAY_SEC = 2.0
MAX_ERROR_RECOVERY_ATTEMPTS = 3

# ROS2 Topics and Services Configuration
# Using relative topic names (~/) for proper namespace handling
TOPIC_AUDIO_CHUNK = "~/audio_chunk"
TOPIC_TRANSCRIPTION_RESULT = "~/transcription_result"
TOPIC_LLM_RESPONSE = "~/llm_response"
TOPIC_ASSISTANT_STATE = "~/assistant_state"
TOPIC_VOICE_EVENT = "~/voice_event"
TOPIC_TTS_AUDIO = "~/tts_audio"

# ROS2 Services - Compatible with voice_assistant_msgs service definitions
SERVICE_GET_STATUS = "~/get_status"
SERVICE_RESET_ASSISTANT = "~/reset_assistant"
SERVICE_START_CONVERSATION = "~/start_conversation"
SERVICE_STOP_CONVERSATION = "~/stop_conversation"

# ROS2 QoS Queue Sizes - Optimized for real-time voice processing
TOPIC_QUEUE_SIZE_AUDIO = 10  # Audio requires moderate buffering for smooth streaming
TOPIC_QUEUE_SIZE_STATE = 1  # State changes are immediate, only latest matters
TOPIC_QUEUE_SIZE_EVENTS = 5  # Event history for debugging and monitoring
TOPIC_QUEUE_SIZE_DIAGNOSTICS = 3  # Diagnostic messages for system health

# ESPHome Voice Assistant Feature Flags
# Based on aioesphomeapi VoiceAssistantFeature
FEATURE_VOICE_ASSISTANT = "voice_assistant"
FEATURE_SPEAKER = "speaker"
FEATURE_MICROPHONE = "microphone"
FEATURE_API_AUDIO = "api_audio"
FEATURE_TIMERS = "timers"
FEATURE_ANNOUNCE = "announce"
FEATURE_START_CONVERSATION = "start_conversation"

# Logging Configuration
LOG_LEVEL_DEFAULT = "INFO"
LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Performance Monitoring
PERFORMANCE_LOG_INTERVAL_SEC = 60.0
AUDIO_LATENCY_WARNING_MS = 100.0
PROCESSING_TIME_WARNING_SEC = 5.0

# Device Configuration
DEVICE_INFO_REFRESH_INTERVAL_SEC = 300.0  # 5 minutes
SUPPORTED_ESPHOME_VERSION_MIN = "2023.12.0"


# Error Codes
class ErrorCodes:
    """Standard error codes for voice assistant system."""

    # Connection Errors (1xxx)
    CONNECTION_FAILED = 1001
    CONNECTION_TIMEOUT = 1002
    AUTHENTICATION_FAILED = 1003
    DEVICE_NOT_FOUND = 1004
    INCOMPATIBLE_VERSION = 1005

    # Audio Errors (2xxx)
    AUDIO_STREAM_ERROR = 2001
    AUDIO_FORMAT_UNSUPPORTED = 2002
    AUDIO_BUFFER_OVERFLOW = 2003
    AUDIO_TIMEOUT = 2004

    # Processing Errors (3xxx)
    STT_PROCESSING_FAILED = 3001
    LLM_PROCESSING_FAILED = 3002
    TTS_PROCESSING_FAILED = 3003
    PIPELINE_TIMEOUT = 3004

    # State Machine Errors (4xxx)
    INVALID_STATE_TRANSITION = 4001
    STATE_TIMEOUT = 4002
    FSM_INTERNAL_ERROR = 4003

    # System Errors (5xxx)
    SYSTEM_OVERLOAD = 5001
    MEMORY_ERROR = 5002
    CONFIGURATION_ERROR = 5003
    UNKNOWN_ERROR = 5999


# Default Configuration Templates
DEFAULT_DEVICE_CONFIG = {
    "host": "192.168.1.100",
    "port": DEFAULT_ESPHOME_PORT,
    "password": None,
    "encryption_key": None,
    "device_name": "ESPHome Voice Assistant",
}

DEFAULT_AUDIO_CONFIG = {
    "sample_rate": DEFAULT_SAMPLE_RATE,
    "sample_width": DEFAULT_SAMPLE_WIDTH,
    "channels": DEFAULT_CHANNELS,
    "chunk_size": AUDIO_CHUNK_SIZE,
    "max_duration": MAX_AUDIO_DURATION_SEC,
}

DEFAULT_PROCESSING_CONFIG = {
    "stt_timeout": STT_PROCESSING_TIMEOUT_SEC,
    "llm_timeout": LLM_PROCESSING_TIMEOUT_SEC,
    "tts_timeout": TTS_PROCESSING_TIMEOUT_SEC,
    "pipeline_timeout": PIPELINE_TOTAL_TIMEOUT_SEC,
}

DEFAULT_CONNECTION_CONFIG = {
    "timeout": CONNECTION_TIMEOUT_SEC,
    "heartbeat_interval": HEARTBEAT_INTERVAL_SEC,
    "reconnect_delay": INITIAL_RECONNECT_DELAY_SEC,
    "max_reconnect_delay": MAX_RECONNECT_DELAY_SEC,
    "max_attempts": MAX_RECONNECTION_ATTEMPTS,
}

# ROS2 Node Configuration
NODE_NAME = "voice_assistant_core"
NODE_NAMESPACE = "/voice_assistant"
PARAMETER_UPDATE_RATE_HZ = 1.0

# Debug and Development Settings
DEBUG_SAVE_AUDIO_CHUNKS = False
DEBUG_AUDIO_FILE_PATH = "/tmp/voice_assistant_audio"
DEBUG_LOG_STATE_TRANSITIONS = True
DEBUG_LOG_AUDIO_STATS = False
DEBUG_PERFORMANCE_METRICS = True
