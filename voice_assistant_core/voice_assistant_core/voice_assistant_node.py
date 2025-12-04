"""
Main ROS2 node for voice assistant core.

This module implements the primary ROS2 node that coordinates the finite state
machine, ESPHome client, audio processing, and provides ROS2 interfaces for
communication with other voice assistant packages.
"""

import asyncio
import json
import logging
import os
import time
from threading import Thread
from typing import Any, Dict, Optional

import numpy as np
import rclpy

try:
    import pvcobra

    COBRA_VAD_AVAILABLE = True
except ImportError:
    COBRA_VAD_AVAILABLE = False
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

# Import custom messages from voice_assistant_msgs
from voice_assistant_msgs.msg import (
    AudioChunk as AudioChunkMsg,
    AssistantState as AssistantStateMsg,
    VoiceEvent as VoiceEventMsg,
    TranscriptionResult as TranscriptionResultMsg,
    LLMResponse as LLMResponseMsg,
    TTSAudio as TTSAudioMsg,
)

from .audio import AudioBuffer, AudioBufferError, AudioChunk
from .audio.types import (
    AssistantState,
    ESPHomeDeviceInfo,
    SystemConfig,
    VoiceEventType,
)
from .communication import ESPHomeClientWrapper, ESPHomeConnectionError
from .constants import (
    HEARTBEAT_INTERVAL_SEC,
    NODE_NAME,
    TOPIC_ASSISTANT_STATE,
    TOPIC_AUDIO_CHUNK,
    TOPIC_LLM_RESPONSE,
    TOPIC_QUEUE_SIZE_EVENTS,
    TOPIC_TRANSCRIPTION_RESULT,
    TOPIC_TTS_AUDIO,
    TOPIC_VOICE_EVENT,
    TURN_DETECTOR_CONFIDENCE_THRESHOLD,
    TURN_DETECTOR_CONTEXT_BUFFER_DURATION,
    TURN_DETECTOR_ENABLE_WARMUP,
    TURN_DETECTOR_ENABLED,
    TURN_DETECTOR_LOG_LEVEL,
    TURN_DETECTOR_MIN_TURN_COOLDOWN,
    TURN_DETECTOR_PRE_BUFFER_DURATION,
    TURN_DETECTOR_SILENCE_ANALYSIS_DURATION,
    TURN_DETECTOR_SILENCE_FALLBACK_TIMEOUT,
    TURN_DETECTOR_SPEECH_START_DURATION,
    VAD_FRAME_LENGTH,
    VAD_THRESHOLD,
    ErrorCodes,
)
from .detection import TurnDetector
from .state import VoiceAssistantStateMachine


class VoiceAssistantNode(Node):
    """
    Main ROS2 node for voice assistant core system.

    Coordinates all voice assistant components including state machine,
    ESPHome communication, audio processing, and ROS2 interfaces.
    """

    def __init__(self):
        """Initialize the voice assistant node."""
        super().__init__(NODE_NAME)

        # Setup logging
        self._setup_logging()

        # Core components
        self._state_machine: Optional[VoiceAssistantStateMachine] = None
        self._esphome_client: Optional[ESPHomeClientWrapper] = None
        self._audio_buffer: Optional[AudioBuffer] = None

        # Voice Activity Detection
        self._vad_engine = None
        self._vad_enabled = False
        self._vad_threshold = VAD_THRESHOLD
        self._vad_silence_counter = 0
        self._vad_is_speaking = False  # Track if user is actively speaking (VAD)

        # Turn Detection - Two-phase system
        self._turn_detector: Optional[TurnDetector] = None
        self._turn_detector_enabled = TURN_DETECTOR_ENABLED
        self._turn_completed_flag = False  # Flag to prevent multiple completions
        self._turn_detector_speech_start_duration = TURN_DETECTOR_SPEECH_START_DURATION
        self._turn_detector_silence_analysis_duration = (
            TURN_DETECTOR_SILENCE_ANALYSIS_DURATION
        )
        self._turn_detector_silence_fallback_timeout = (
            TURN_DETECTOR_SILENCE_FALLBACK_TIMEOUT
        )
        self._turn_detector_context_buffer_duration = (
            TURN_DETECTOR_CONTEXT_BUFFER_DURATION
        )
        self._turn_detector_pre_buffer_duration = TURN_DETECTOR_PRE_BUFFER_DURATION
        self._turn_detector_confidence_threshold = TURN_DETECTOR_CONFIDENCE_THRESHOLD
        self._turn_detector_min_turn_cooldown = TURN_DETECTOR_MIN_TURN_COOLDOWN
        self._turn_detector_enable_warmup = TURN_DETECTOR_ENABLE_WARMUP
        self._turn_detector_log_level = TURN_DETECTOR_LOG_LEVEL

        # VAD state tracking for turn detection coordination
        self._previous_vad_state = "quiet"  # Track VAD state transitions

        # ROS2 callback groups
        self._default_cb_group = ReentrantCallbackGroup()
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()

        # Configuration
        self._system_config: Optional[SystemConfig] = None

        # Async event loop management
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[Thread] = None
        self._shutdown_event = asyncio.Event()

        # ROS2 interface setup
        self._setup_parameters()
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_services()
        self._setup_timers()

        # State tracking
        self._last_state_publish = 0.0
        self._last_heartbeat = 0.0
        self._initialization_complete = False

        self.get_logger().info("Voice Assistant Node initialized")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        )

        # Set logger for this class
        self._logger = self.get_logger()

    def _setup_parameters(self) -> None:
        """Setup ROS2 parameters."""
        # Device connection parameters
        self.declare_parameter("device.host", "192.168.1.100")
        self.declare_parameter("device.port", 6053)
        self.declare_parameter("device.password", "")
        self.declare_parameter("device.encryption_key", "")
        self.declare_parameter("device.name", "ESPHome Voice Assistant")

        # Audio parameters
        self.declare_parameter("audio.sample_rate", 16000)
        self.declare_parameter("audio.sample_width", 16)
        self.declare_parameter("audio.channels", 1)
        self.declare_parameter("audio.max_duration", 30.0)
        self.declare_parameter("audio.buffer_size", 1024 * 1024)

        # Voice Activity Detection (VAD) parameters
        # VAD parameters
        self.declare_parameter("vad.enabled", True)
        self.declare_parameter("vad.access_key", os.getenv("VAD_ACCESS_KEY", ""))
        self.declare_parameter("vad.threshold", VAD_THRESHOLD)
        self.declare_parameter("vad.min_silence_frames", 30)
        self.declare_parameter("vad.stop_secs", 1.0)

        # Audio buffer parameters
        self.declare_parameter("audio.log_level", "WARNING")

        # Turn Detection parameters - Two-phase system
        self.declare_parameter("turn_detector.enabled", TURN_DETECTOR_ENABLED)
        self.declare_parameter(
            "turn_detector.speech_start_consecutive_duration",
            TURN_DETECTOR_SPEECH_START_DURATION,
        )
        self.declare_parameter(
            "turn_detector.silence_duration_before_analysis",
            TURN_DETECTOR_SILENCE_ANALYSIS_DURATION,
        )
        self.declare_parameter(
            "turn_detector.silence_fallback_timeout",
            TURN_DETECTOR_SILENCE_FALLBACK_TIMEOUT,
        )
        self.declare_parameter(
            "turn_detector.context_buffer_duration",
            TURN_DETECTOR_CONTEXT_BUFFER_DURATION,
        )
        self.declare_parameter(
            "turn_detector.pre_buffer_duration", TURN_DETECTOR_PRE_BUFFER_DURATION
        )
        self.declare_parameter(
            "turn_detector.ml_confidence_threshold", TURN_DETECTOR_CONFIDENCE_THRESHOLD
        )
        self.declare_parameter(
            "turn_detector.min_turn_cooldown", TURN_DETECTOR_MIN_TURN_COOLDOWN
        )
        self.declare_parameter(
            "turn_detector.enable_model_warmup", TURN_DETECTOR_ENABLE_WARMUP
        )
        self.declare_parameter("turn_detector.log_level", TURN_DETECTOR_LOG_LEVEL)

        # Connection parameters
        self.declare_parameter("connection.timeout", 10.0)
        self.declare_parameter("connection.heartbeat_interval", 30.0)
        self.declare_parameter("connection.auto_reconnect", True)

        # Debug parameters
        self.declare_parameter("debug.enable_logging", False)
        self.declare_parameter("debug.save_audio", False)

        self._logger.info("Parameters declared")

    def _setup_publishers(self) -> None:
        """Setup ROS2 publishers."""
        # Optimized QoS profiles - reduced queue sizes for better performance
        audio_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,  # Reduced from TOPIC_QUEUE_SIZE_AUDIO for lower latency
        )

        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=3,  # Reduced from TOPIC_QUEUE_SIZE_STATE - only current state matters
        )

        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # Changed to RELIABLE for consistency
            history=HistoryPolicy.KEEP_LAST,
            depth=5,  # Reduced queue size
        )

        # Publishers - use custom messages for optimal performance
        self._audio_chunk_pub = self.create_publisher(
            AudioChunkMsg, TOPIC_AUDIO_CHUNK, audio_qos, callback_group=self._default_cb_group
        )
        self._state_pub = self.create_publisher(
            AssistantStateMsg,
            TOPIC_ASSISTANT_STATE,
            state_qos,
            callback_group=self._default_cb_group,
        )
        self._event_pub = self.create_publisher(
            VoiceEventMsg, TOPIC_VOICE_EVENT, event_qos, callback_group=self._default_cb_group
        )

        self._logger.info("Publishers created")

    def _setup_subscribers(self) -> None:
        """Setup ROS2 subscribers."""
        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=TOPIC_QUEUE_SIZE_EVENTS,
        )

        # Subscribers - use custom messages for optimal performance
        self._transcription_sub = self.create_subscription(
            TranscriptionResultMsg,
            TOPIC_TRANSCRIPTION_RESULT,
            self._transcription_callback,
            event_qos,
            callback_group=self._default_cb_group,
        )

        self._llm_response_sub = self.create_subscription(
            LLMResponseMsg,
            TOPIC_LLM_RESPONSE,
            self._llm_response_callback,
            event_qos,
            callback_group=self._default_cb_group,
        )

        self._tts_audio_sub = self.create_subscription(
            TTSAudioMsg,
            TOPIC_TTS_AUDIO,
            self._tts_audio_callback,
            event_qos,
            callback_group=self._default_cb_group,
        )

        self._logger.info("Subscribers created")

    def _setup_services(self) -> None:
        """Setup ROS2 services."""
        # Services will be implemented when needed
        # For now, we'll use topics for communication
        self._logger.info("Services setup complete")

    def _setup_timers(self) -> None:
        """Setup ROS2 timers."""
        # State publisher timer - reduced frequency for better performance
        self._state_timer = self.create_timer(
            5.0,  # Reduced from 1.0 Hz to 0.2 Hz - state doesn't change that often
            self._state_timer_callback,
            callback_group=self._timer_cb_group,
        )

        # Heartbeat timer - increased interval
        self._heartbeat_timer = self.create_timer(
            HEARTBEAT_INTERVAL_SEC * 2,  # Doubled interval to reduce overhead
            self._heartbeat_callback,
            callback_group=self._timer_cb_group,
        )

        # Diagnostic timer - disabled in production builds
        # Only enable when needed for debugging
        # self._diagnostic_timer = self.create_timer(
        #     60.0,  # Increased to 60 seconds if needed
        #     self._diagnostic_callback,
        #     callback_group=self._timer_cb_group,
        # )

        self._logger.info("Timers created")

    def _setup_vad(self) -> None:
        """Setup Voice Activity Detection."""
        try:
            # Get VAD parameters
            self._vad_enabled = (
                self.get_parameter("vad.enabled").get_parameter_value().bool_value
            )
            vad_access_key = (
                self.get_parameter("vad.access_key").get_parameter_value().string_value
            )
            self._vad_threshold = (
                self.get_parameter("vad.threshold").get_parameter_value().double_value
            )
            self._vad_min_silence_frames = (
                self.get_parameter("vad.min_silence_frames")
                .get_parameter_value()
                .integer_value
            )
            self._vad_stop_secs = (
                self.get_parameter("vad.stop_secs").get_parameter_value().double_value
            )

            if not self._vad_enabled:
                self._logger.info("VAD is disabled")
                return

            if not COBRA_VAD_AVAILABLE:
                self._logger.warning("pvcobra is not available, VAD disabled")
                self._vad_enabled = False
                return

            if not vad_access_key:
                self._logger.warning("VAD access key is empty, VAD disabled")
                self._vad_enabled = False
                return

            # Initialize Cobra VAD
            self._vad_engine = pvcobra.create(access_key=vad_access_key)
            self._logger.info(f"VAD initialized with threshold: {self._vad_threshold}")

        except Exception as e:
            self._logger.error(f"Failed to initialize VAD: {e}")
            self._vad_enabled = False
            self._vad_engine = None

    def _setup_turn_detector(self) -> None:
        """Setup two-phase turn detector."""
        try:
            # Get turn detector parameters
            self._turn_detector_enabled = (
                self.get_parameter("turn_detector.enabled")
                .get_parameter_value()
                .bool_value
            )
            self._turn_detector_speech_start_duration = (
                self.get_parameter("turn_detector.speech_start_consecutive_duration")
                .get_parameter_value()
                .double_value
            )
            self._turn_detector_silence_analysis_duration = (
                self.get_parameter("turn_detector.silence_duration_before_analysis")
                .get_parameter_value()
                .double_value
            )
            self._turn_detector_silence_fallback_timeout = (
                self.get_parameter("turn_detector.silence_fallback_timeout")
                .get_parameter_value()
                .double_value
            )
            self._turn_detector_context_buffer_duration = (
                self.get_parameter("turn_detector.context_buffer_duration")
                .get_parameter_value()
                .double_value
            )
            self._turn_detector_pre_buffer_duration = (
                self.get_parameter("turn_detector.pre_buffer_duration")
                .get_parameter_value()
                .double_value
            )
            self._turn_detector_confidence_threshold = (
                self.get_parameter("turn_detector.ml_confidence_threshold")
                .get_parameter_value()
                .double_value
            )
            self._turn_detector_min_turn_cooldown = (
                self.get_parameter("turn_detector.min_turn_cooldown")
                .get_parameter_value()
                .double_value
            )
            self._turn_detector_enable_warmup = (
                self.get_parameter("turn_detector.enable_model_warmup")
                .get_parameter_value()
                .bool_value
            )
            self._turn_detector_log_level = (
                self.get_parameter("turn_detector.log_level")
                .get_parameter_value()
                .string_value
            )

            if not self._turn_detector_enabled:
                self._logger.info("Turn detector is disabled")
                return

            # Get chunk size from audio frame (Cobra VAD uses 512 samples)
            chunk_size = VAD_FRAME_LENGTH

            # Initialize two-phase turn detector
            self._turn_detector = TurnDetector(
                speech_start_consecutive_duration=self._turn_detector_speech_start_duration,
                silence_duration_before_analysis=self._turn_detector_silence_analysis_duration,
                silence_fallback_timeout=self._turn_detector_silence_fallback_timeout,
                context_buffer_duration=self._turn_detector_context_buffer_duration,
                pre_buffer_duration=self._turn_detector_pre_buffer_duration,
                ml_confidence_threshold=self._turn_detector_confidence_threshold,
                min_turn_cooldown=self._turn_detector_min_turn_cooldown,
                sample_rate=16000,
                chunk_size=chunk_size,
                enabled=self._turn_detector_enabled,
                enable_model_warmup=self._turn_detector_enable_warmup,
                log_level=self._turn_detector_log_level,
            )

            if self._turn_detector.enabled:
                self._logger.info(
                    f"Turn detector initialized (two-phase): "
                    f"speech_start={self._turn_detector_speech_start_duration}s, "
                    f"silence_analysis={self._turn_detector_silence_analysis_duration}s, "
                    f"fallback={self._turn_detector_silence_fallback_timeout}s, "
                    f"ml_threshold={self._turn_detector_confidence_threshold}"
                )

                # Run model warmup if enabled
                if self._turn_detector_enable_warmup:
                    asyncio.create_task(self._warmup_turn_detector())
            else:
                self._logger.warning("Turn detector initialization failed")

        except Exception as e:
            self._logger.error(f"Failed to initialize turn detector: {e}")
            self._turn_detector_enabled = False
            self._turn_detector = None

    async def _warmup_turn_detector(self) -> None:
        """Warm up turn detector model asynchronously."""
        try:
            if self._turn_detector:
                await self._turn_detector.warm_up()
        except Exception as e:
            self._logger.warning(f"Turn detector warmup failed: {e}")

    async def async_init(self) -> bool:
        """
        Async initialization of core components.

        Returns:
            True if initialization successful
        """
        try:
            self._logger.info("Starting async initialization")

            # Load configuration
            await self._load_configuration()

            # Initialize state machine
            self._state_machine = VoiceAssistantStateMachine(
                AssistantState.DISCONNECTED
            )
            self._setup_state_machine_callbacks()

            # Initialize audio buffer
            audio_log_level = (
                self.get_parameter("audio.log_level").get_parameter_value().string_value
            )
            self._audio_buffer = AudioBuffer(
                max_buffer_size=self._system_config.audio_buffer_size,
                max_duration=self._system_config.max_audio_duration,
                log_level=audio_log_level,
            )
            self._setup_audio_buffer_callbacks()

            # Initialize VAD and Turn Detector (requires system_config to be loaded)
            self._setup_vad()
            self._setup_turn_detector()

            # Initialize ESPHome client
            self._esphome_client = ESPHomeClientWrapper(self._system_config.device_info)
            self._setup_esphome_callbacks()

            # Start connection process
            asyncio.create_task(self._connection_manager())

            self._initialization_complete = True
            self._logger.info("Async initialization complete")

            return True

        except Exception as e:
            self._logger.error(f"Async initialization failed: {e}")
            return False

    async def _load_configuration(self) -> None:
        """Load system configuration from parameters."""
        # Device configuration
        device_info = ESPHomeDeviceInfo(
            host=self.get_parameter("device.host").value,
            port=self.get_parameter("device.port").value,
            password=self.get_parameter("device.password").value or None,
            encryption_key=self.get_parameter("device.encryption_key").value or None,
            device_name=self.get_parameter("device.name").value,
        )

        # System configuration
        self._system_config = SystemConfig(
            device_info=device_info,
            audio_buffer_size=self.get_parameter("audio.buffer_size").value,
            connection_timeout=self.get_parameter("connection.timeout").value,
            heartbeat_interval=self.get_parameter(
                "connection.heartbeat_interval"
            ).value,
            max_audio_duration=self.get_parameter("audio.max_duration").value,
            enable_debug_logging=self.get_parameter("debug.enable_logging").value,
        )

        self._logger.info(
            f"Configuration loaded for device: {device_info.host}:{device_info.port}"
        )

    def _setup_state_machine_callbacks(self) -> None:
        """Setup state machine event callbacks."""
        if not self._state_machine:
            return

        # Register state entry callbacks
        self._state_machine.register_state_entry_callback(
            AssistantState.CONNECTING, self._on_connecting_state
        )
        self._state_machine.register_state_entry_callback(
            AssistantState.IDLE, self._on_idle_state
        )
        self._state_machine.register_state_entry_callback(
            AssistantState.STREAMING_AUDIO, self._on_streaming_state
        )
        self._state_machine.register_state_entry_callback(
            AssistantState.ERROR, self._on_error_state
        )

        self._logger.info("State machine callbacks configured")

    def _setup_audio_buffer_callbacks(self) -> None:
        """Setup audio buffer event callbacks."""
        if not self._audio_buffer:
            return

        self._audio_buffer.set_stream_start_callback(self._on_audio_stream_start)
        self._audio_buffer.set_stream_end_callback(self._on_audio_stream_end)
        self._audio_buffer.set_chunk_callback(self._on_audio_chunk)

        self._logger.info("Audio buffer callbacks configured")

    def _setup_esphome_callbacks(self) -> None:
        """Setup ESPHome client event callbacks with enhanced validation."""
        if not self._esphome_client:
            return

        self._esphome_client.set_pipeline_start_callback(self._on_pipeline_start)
        self._esphome_client.set_pipeline_stop_callback(self._on_pipeline_stop)
        self._esphome_client.set_audio_callback(self._on_esphome_audio)
        self._esphome_client.set_connection_callback(self._on_connection_change)

        self._logger.info("ESPHome client callbacks configured")
        self._logger.debug(
            f"Callback registration status: {self._esphome_client.get_status()['voice_assistant']['callbacks_registered']}"
        )

    async def _connection_manager(self) -> None:
        """Manage ESPHome device connection with enhanced monitoring."""
        connection_attempt = 0
        while not self._shutdown_event.is_set():
            try:
                if not self._esphome_client.is_connected:
                    connection_attempt += 1
                    self._logger.info(f"Connection attempt #{connection_attempt}")

                    # Transition to connecting state
                    if self._state_machine.current_state == AssistantState.DISCONNECTED:
                        self._state_machine.transition_to(AssistantState.CONNECTING)

                    # Attempt connection
                    success = await self._esphome_client.connect()

                    if success:
                        # Validate subscription after connection
                        validation = self._esphome_client.validate_subscription_state()
                        if validation["is_valid"]:
                            self._logger.info(
                                "Voice assistant subscription validated successfully"
                            )
                            # Transition to idle state
                            self._state_machine.transition_to(AssistantState.IDLE)
                            connection_attempt = 0  # Reset counter on success
                        else:
                            self._logger.error(
                                f"Subscription validation failed: {validation['issues']}"
                            )
                            # Force disconnection and retry
                            await self._esphome_client.disconnect()
                            await asyncio.sleep(5.0)
                            continue

                else:
                    # Already connected - periodic health check
                    status = self._esphome_client.get_status()
                    if status["voice_assistant"]["subscription_active"]:
                        self._logger.debug(
                            "Voice assistant subscription health check: OK"
                        )
                    else:
                        self._logger.warning(
                            "Voice assistant subscription health check: FAILED"
                        )
                        # Attempt to reconnect
                        await self._esphome_client.disconnect()
                        continue

                # Wait before next check
                await asyncio.sleep(10.0 if self._esphome_client.is_connected else 5.0)

            except ESPHomeConnectionError as e:
                self._logger.error(
                    f"Connection error (attempt #{connection_attempt}): {e}"
                )
                self._state_machine.handle_error(ErrorCodes.CONNECTION_FAILED, str(e))
                # Exponential backoff with max 60 seconds
                wait_time = min(10.0 * (1.5 ** min(connection_attempt - 1, 5)), 60.0)
                await asyncio.sleep(wait_time)

            except Exception as e:
                self._logger.error(
                    f"Unexpected connection manager error: {e}", exc_info=True
                )
                await asyncio.sleep(5.0)

    # State machine callbacks
    async def _on_connecting_state(self, prev_state, curr_state, event_data):
        """Handle entering CONNECTING state."""
        self._publish_event(
            VoiceEventType.CONNECTION_ESTABLISHED,
            {"previous_state": prev_state.name if prev_state else None},
        )

    async def _on_idle_state(self, prev_state, curr_state, event_data):
        """Handle entering IDLE state."""
        self._logger.info("Voice assistant ready and waiting for wake word")
        self._publish_event(
            VoiceEventType.CONNECTION_ESTABLISHED,
            {"previous_state": prev_state.name if prev_state else None},
        )

    async def _on_streaming_state(self, prev_state, curr_state, event_data):
        """Handle entering STREAMING_AUDIO state."""
        self._logger.info("Audio streaming started")
        self._publish_event(VoiceEventType.AUDIO_STREAM_START, event_data or {})

    async def _on_error_state(self, prev_state, curr_state, event_data):
        """Handle entering ERROR state."""
        self._logger.error(
            f"Entered error state from {prev_state.name if prev_state else 'None'}"
        )
        self._publish_event(VoiceEventType.ERROR_OCCURRED, event_data or {})

        # Schedule recovery attempt
        asyncio.create_task(self._schedule_error_recovery())

    async def _schedule_error_recovery(self) -> None:
        """Schedule error recovery attempt."""
        await asyncio.sleep(5.0)  # Wait before recovery

        if self._state_machine.current_state == AssistantState.ERROR:
            self._state_machine.attempt_recovery()

    # Audio callbacks
    def _on_audio_stream_start(self, stream_id, stats):
        """Handle audio stream start (sync callback)."""
        # Schedule async handler in the event loop
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                self._async_audio_stream_start(stream_id, stats), self._loop
            )

    async def _async_audio_stream_start(self, stream_id, stats):
        """Async handler for audio stream start.

        Implements the Pipecat pattern so the stream can start or restart from multiple states
        to support continuous conversation and interruptions.
        """
        try:
            self._logger.info(
                f"Audio stream started - ID: {stream_id}, state: {self._state_machine.current_state.name}"
            )

            # If a processing state receives a new stream, treat it as a user interruption or additional information
            if self._state_machine.current_state in [
                AssistantState.TRANSCRIBING,
                AssistantState.PROCESSING_LLM,
                AssistantState.PLAYING_RESPONSE,
            ]:
                self._logger.info(
                    f"New audio stream during {self._state_machine.current_state.name} - "
                    "usuario continúa interacción"
                )
                # The transition will be handled by _on_esphome_audio once the first chunk arrives

        except Exception as e:
            self._logger.error(f"Error handling audio stream start: {e}")

    def _on_audio_stream_end(self, stream_id, stats):
        """Handle audio stream end (sync callback)."""
        # Schedule async handler in the event loop with error handling
        if self._loop and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._async_audio_stream_end(stream_id, stats), self._loop
                )
            except RuntimeError:
                # Event loop closed or not available
                self._logger.warning(
                    "Cannot schedule audio stream end - event loop unavailable"
                )

    async def _async_audio_stream_end(self, stream_id, stats):
        """
        Async handler for audio stream end.

        Note: With the new two-phase turn detector, this callback is rarely used
        as turn completion is handled directly in _process_audio_with_vad.
        """
        try:
            self._logger.info(
                f"Audio stream ended - ID: {stream_id}, Duration: {stats.stream_duration:.2f}s"
            )

            # Get audio from turn detector's context buffer
            if self._turn_detector and self._turn_detector_enabled:
                audio_array = self._turn_detector.get_audio_for_transcription()
                if audio_array is not None:
                    # Convert float32 numpy array to bytes (int16)
                    audio_int16 = (audio_array * 32768.0).astype(np.int16)
                    audio_data = audio_int16.tobytes()

                    if len(audio_data) > 0:
                        self._publish_audio_for_transcription(audio_data)

                        # Transition to transcribing if in streaming state
                        if (
                            self._state_machine.current_state
                            == AssistantState.STREAMING_AUDIO
                        ):
                            self._state_machine.transition_to(
                                AssistantState.TRANSCRIBING
                            )

        except Exception as e:
            self._logger.error(f"Error handling audio stream end: {e}")

    def _on_audio_chunk(self, chunk, stats):
        """Handle new audio chunk (sync callback)."""
        # Removed debug logging for performance - only log critical issues
        # Schedule async publication in the event loop with error handling
        if self._loop and not self._loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(
                    self._async_publish_audio_chunk(chunk), self._loop
                )
            except RuntimeError:
                # Event loop closed - silently drop chunk to avoid log spam
                pass
        else:
            # Only log once to avoid spam
            if not hasattr(self, "_loop_warning_logged"):
                self._logger.warning(
                    "Event loop not available for audio chunk publication"
                )
                self._loop_warning_logged = True

    async def _async_publish_audio_chunk(self, chunk):
        """Async handler for audio chunk publication with VAD filtering."""
        try:
            # Only publish audio chunks if we're in streaming state
            # CRITICAL: Return early to prevent auto-starting during TRANSCRIBING
            if self._state_machine.current_state != AssistantState.STREAMING_AUDIO:
                # Silently drop chunks when not streaming (normal during state transitions)
                return

            # Apply VAD filtering before publishing
            if await self._process_audio_with_vad(chunk):
                self._publish_audio_chunk(chunk)
            else:
                # Check if we should stop the audio stream due to prolonged silence
                if (
                    self._vad_silence_counter >= self._vad_min_silence_frames
                    and self._state_machine
                    and self._state_machine.current_state
                    == AssistantState.STREAMING_AUDIO
                ):
                    self._logger.debug(
                        f"Prolonged silence detected ({self._vad_silence_counter} frames), considering end of speech"
                    )
                    # Note: We don't automatically stop here to avoid cutting off speech
                    # The ESPHome device will handle the stream end based on its own logic
        except Exception as e:
            self._logger.error(f"Error publishing audio chunk: {e}")

    # ESPHome callbacks
    async def _on_pipeline_start(
        self, conversation_id, flags, audio_settings, wake_word_phrase
    ):
        """Handle pipeline start from ESPHome device."""
        self._logger.info(f"Pipeline started - Wake word: {wake_word_phrase}")

        # Transition to streaming state
        if self._state_machine.current_state == AssistantState.IDLE:
            self._state_machine.transition_to(AssistantState.STREAMING_AUDIO)

            # Extract audio settings safely (following HA pattern)
            sample_rate = 16000  # Default
            sample_width = 16  # Default 16-bit
            channels = 1  # Default mono

            if audio_settings:
                # Try to get sample rate from audio settings
                sample_rate = getattr(audio_settings, "sample_rate", 16000)
                # Some ESPHome versions might use different attribute names
                if sample_rate == 16000 and hasattr(audio_settings, "sample_frequency"):
                    sample_rate = getattr(audio_settings, "sample_frequency", 16000)

                # Get other audio parameters if available
                sample_width = getattr(audio_settings, "sample_width", 16)
                if sample_width == 16 and hasattr(audio_settings, "bits_per_sample"):
                    sample_width = getattr(audio_settings, "bits_per_sample", 16)

                channels = getattr(audio_settings, "channels", 1)
                if channels == 1 and hasattr(audio_settings, "num_channels"):
                    channels = getattr(audio_settings, "num_channels", 1)

                self._logger.debug(
                    f"Audio settings: {sample_rate}Hz, {sample_width}-bit, {channels} channels"
                )
            else:
                self._logger.debug("No audio settings provided, using defaults")

            # Start audio stream with extracted settings
            self._audio_buffer.start_stream(
                sample_rate=sample_rate, sample_width=sample_width, channels=channels
            )

            # Reset turn detector for new conversation
            if self._turn_detector:
                self._turn_detector.reset_stream()

            # Reset turn completion flag
            self._turn_completed_flag = False

        return 0  # Use API audio mode

    async def _on_pipeline_stop(self, abort):
        """Handle pipeline stop from ESPHome device."""
        self._logger.info(f"Pipeline stopped - Abort: {abort}")

        # End audio stream
        if self._audio_buffer.is_streaming:
            self._audio_buffer.end_stream()

        # Return to idle if not in error state
        if self._state_machine.current_state not in [
            AssistantState.ERROR,
            AssistantState.IDLE,
        ]:
            self._state_machine.transition_to(AssistantState.IDLE)

    async def _on_esphome_audio(self, data):
        """Handle audio data from ESPHome device.

        Implementation inspired by the Home Assistant assist_pipeline:
        - Audio flows continuously without interruption
        - Audio is not rejected during TRANSCRIBING or PROCESSING
        - New audio indicates the user has more to say
        """
        self._logger.debug(
            f"ESPHome audio received: {len(data)} bytes, streaming: {self._audio_buffer.is_streaming}, state: {self._state_machine.current_state.name}"
        )
        try:
            # Case 1: Stream already active - append the chunk directly
            if self._audio_buffer.is_streaming:
                added = self._audio_buffer.add_chunk(data)
                self._logger.debug(f"Audio chunk added to buffer: {added}")

                # Pipecat pattern: when audio arrives during TRANSCRIBING, determine whether
                # it represents actual user speech (VAD SPEAKING) or residual chunks.
                # Cancel the transcription only if VAD SPEAKING is confirmed.
                if self._state_machine.current_state == AssistantState.TRANSCRIBING:
                    # If genuine VAD SPEAKING activity is detected, treat it as a continuation
                    if self._vad_is_speaking:
                        self._logger.info(
                            "Usuario continúa hablando después de boundary (VAD SPEAKING) - "
                            "falso positivo del modelo, continuando turn"
                        )
                        # Return to STREAMING_AUDIO since the user remains in their turn
                        self._state_machine.transition_to(
                            AssistantState.STREAMING_AUDIO
                        )
                        self._boundary_marked = False  # Clear boundary flag
                        self._publish_event(
                            VoiceEventType.AUDIO_STREAM_START,
                            {
                                "reason": "false_positive_continuation",
                                "previous_state": "TRANSCRIBING",
                            },
                        )
                    else:
                        # These are just residual chunks in transit - ignore them
                        self._logger.debug(
                            "Chunks residuales durante TRANSCRIBING - ignorando (no hay VAD SPEAKING)"
                        )
                    return

                # When in PROCESSING_LLM, new audio means the user intends to interrupt
                if self._state_machine.current_state == AssistantState.PROCESSING_LLM:
                    self._logger.info(
                        "Usuario habla durante PROCESSING_LLM - interrumpiendo"
                    )
                    self._state_machine.transition_to(AssistantState.STREAMING_AUDIO)
                    # Reset the turn detector to begin a clean new turn
                    if self._turn_detector and self._turn_detector_enabled:
                        self._turn_detector.reset_stream()
                    self._publish_event(
                        VoiceEventType.AUDIO_STREAM_START,
                        {
                            "reason": "user_interruption",
                            "previous_state": "PROCESSING_LLM",
                        },
                    )
                    return

                # When the user speaks during PLAYING_RESPONSE, always interrupt
                if self._state_machine.current_state == AssistantState.PLAYING_RESPONSE:
                    self._logger.info("Usuario interrumpe durante PLAYING_RESPONSE")
                    self._state_machine.transition_to(AssistantState.STREAMING_AUDIO)
                    # Reset the turn detector for the next turn
                    if self._turn_detector and self._turn_detector_enabled:
                        self._turn_detector.reset_stream()
                    self._publish_event(
                        VoiceEventType.AUDIO_STREAM_START,
                        {
                            "reason": "user_interruption",
                            "previous_state": "PLAYING_RESPONSE",
                        },
                    )
                return  # Case 2: Stream inactive - start it only from an appropriate state
            if self._state_machine.current_state in [
                AssistantState.IDLE,
                AssistantState.STREAMING_AUDIO,  # Restart if it was already streaming
            ]:
                self._logger.info("Starting audio stream on first audio data")
                try:
                    # Get audio parameters from config
                    sample_rate = self.get_parameter("audio.sample_rate").value
                    sample_width = self.get_parameter("audio.sample_width").value
                    channels = self.get_parameter("audio.channels").value

                    # Start audio stream
                    self._audio_buffer.start_stream(
                        sample_rate=sample_rate,
                        sample_width=sample_width,
                        channels=channels,
                    )

                    # Transition to streaming state
                    self._state_machine.transition_to(AssistantState.STREAMING_AUDIO)

                    # Add the chunk
                    added = self._audio_buffer.add_chunk(data)
                    self._logger.debug(f"Audio chunk added after stream start: {added}")

                except Exception as start_error:
                    self._logger.error(f"Failed to start audio stream: {start_error}")
            else:
                # Stream inactive and not in an appropriate state - buffer the audio quietly
                # This can occur during state transitions
                self._logger.debug(
                    f"Buffering audio chunk in state {self._state_machine.current_state.name} "
                    "(stream will start on next pipeline run)"
                )

        except AudioBufferError as e:
            self._logger.error(f"Audio buffer error: {e}")
            self._state_machine.handle_error(ErrorCodes.AUDIO_BUFFER_OVERFLOW, str(e))

    async def _on_connection_change(self, connected, device_info):
        """Handle ESPHome connection state change."""
        if connected:
            self._logger.info("ESPHome device connected")
            if self._state_machine.current_state == AssistantState.CONNECTING:
                self._state_machine.transition_to(AssistantState.IDLE)
        else:
            self._logger.warning("ESPHome device disconnected")
            if self._state_machine.current_state != AssistantState.ERROR:
                self._state_machine.transition_to(AssistantState.DISCONNECTED)

    # ROS2 callbacks
    def _transcription_callback(self, msg: TranscriptionResultMsg) -> None:
        """Handle transcription result."""
        try:
            text = msg.text
            confidence = msg.confidence

            # Only log if text is significant (>5 chars) to reduce noise
            if len(text) > 5:
                self._logger.info(f"Transcription [FINAL]: {text[:50]}... (conf: {confidence:.2f})")

            # OPTIMIZATION: Transition to LLM processing immediately on final transcription
            # Don't wait for additional messages - process as soon as we have final text
            if self._state_machine.current_state == AssistantState.TRANSCRIBING:
                self._logger.info(
                    "Final transcription received - transitioning to LLM processing"
                )
                self._state_machine.transition_to(AssistantState.PROCESSING_LLM)

        except Exception as e:
            self._logger.error(f"Error processing transcription message: {e}")

    def _llm_response_callback(self, msg: LLMResponseMsg) -> None:
        """Handle LLM response."""
        try:
            response_text = msg.response_text
            confidence = msg.confidence
            
            self._logger.info(f"LLM response received (conf: {confidence:.2f}): {response_text[:50]}...")

            # Transition to TTS processing state
            if self._state_machine.current_state == AssistantState.PROCESSING_LLM:
                self._state_machine.transition_to(AssistantState.GENERATING_SPEECH)

        except Exception as e:
            self._logger.error(f"Error processing LLM response message: {e}")

    def _tts_audio_callback(self, msg: TTSAudioMsg) -> None:
        """Handle TTS audio for playback."""
        try:
            audio_bytes = bytes(msg.audio_data)
            self._logger.info(f"TTS audio received for playback: {len(audio_bytes)} bytes")

            # Send audio to ESPHome device
            if self._esphome_client and self._esphome_client.is_connected:
                asyncio.create_task(
                    self._esphome_client.send_voice_assistant_audio(audio_bytes)
                )

            # Return to idle state after playback
            if self._state_machine.current_state == AssistantState.PLAYING_RESPONSE:
                self._state_machine.transition_to(AssistantState.IDLE)

        except Exception as e:
            self._logger.error(f"Error processing TTS audio message: {e}")

    def _state_timer_callback(self) -> None:
        """Periodic state publishing."""
        current_time = time.time()

        # Publish state only when needed - optimized for performance
        if (current_time - self._last_state_publish) > 5.0:  # Reduced frequency
            self._publish_current_state()
            self._last_state_publish = current_time

    def _heartbeat_callback(self) -> None:
        """Heartbeat timer callback."""
        current_time = time.time()
        self._last_heartbeat = current_time

        # Publish heartbeat event
        self._publish_event(
            VoiceEventType.HEARTBEAT,
            {
                "heartbeat_time": current_time,
                "uptime": current_time - self._last_state_publish
                if self._last_state_publish > 0
                else 0,
            },
        )

    def _diagnostic_callback(self) -> None:
        """Diagnostic timer callback for debugging subscription issues."""
        if not self._initialization_complete or not self._esphome_client:
            return

        try:
            # Get comprehensive status
            status = self._esphome_client.get_status()
            validation = self._esphome_client.validate_subscription_state()

            # Log diagnostic information
            self._logger.info("=== VOICE ASSISTANT DIAGNOSTIC ===")
            self._logger.info(
                f"Connection: {'OK' if status['is_connected'] else 'FAILED'}"
            )
            self._logger.info(
                f"Subscription: {'ACTIVE' if status['voice_assistant']['subscription_active'] else 'INACTIVE'}"
            )
            self._logger.info(f"Features: {status['voice_assistant']['features']}")

            callbacks = status["voice_assistant"]["callbacks_registered"]
            self._logger.info(
                f"Callbacks - Start: {callbacks['pipeline_start']}, "
                f"Stop: {callbacks['pipeline_stop']}, "
                f"Audio: {callbacks['audio']}, "
                f"Connection: {callbacks['connection']}"
            )

            if not validation["is_valid"]:
                self._logger.warning(f"Validation issues: {validation['issues']}")
                for rec in validation["recommendations"]:
                    self._logger.info(f"Recommendation: {rec}")

            # Publish diagnostic event
            self._publish_event(
                VoiceEventType.HEARTBEAT,
                {
                    "diagnostic_type": "voice_assistant_status",
                    **status,
                    "validation": validation,
                },
            )

            self._logger.info("=== END DIAGNOSTIC ===")

        except Exception as e:
            self._logger.error(f"Error in diagnostic callback: {e}", exc_info=True)

    # Publishing methods
    def _publish_current_state(self) -> None:
        """Publish current assistant state."""
        if not self._state_machine:
            return

        state_msg = AssistantStateMsg()
        state_msg.current_state = self._state_machine.current_state.name
        state_msg.previous_state = self._state_machine.previous_state.name if self._state_machine.previous_state else ""
        state_msg.transition_time.sec = int(time.time())
        state_msg.transition_time.nanosec = int((time.time() % 1) * 1e9)
        
        # Add state metadata as JSON
        state_data = {
            "time_in_state": self._state_machine.time_in_current_state,
            "error_count": self._state_machine.error_count,
        }
        state_msg.state_data = json.dumps(state_data)
        
        self._state_pub.publish(state_msg)

    def _publish_event(self, event_type: VoiceEventType, data: Dict[str, Any]) -> None:
        """Publish voice event."""
        event_msg = VoiceEventMsg()
        event_msg.event_type = event_type.name
        event_msg.message = data.get("message", "")
        event_msg.timestamp.sec = int(time.time())
        event_msg.timestamp.nanosec = int((time.time() % 1) * 1e9)
        event_msg.priority = data.get("priority", 0)  # 0=info, 1=warning, 2=error
        event_msg.event_data = json.dumps(data)
        self._event_pub.publish(event_msg)

    async def _process_audio_with_vad(self, chunk: AudioChunk) -> bool:
        """
        Process audio chunk with VAD and two-phase turn detection.

        New two-phase approach:
        1. Phase 1: Wait for consecutive speech burst (handled in TurnDetector)
        2. Phase 2: Analyze during silence + fallback timeout (handled in TurnDetector)

        Args:
            chunk: Audio chunk to process

        Returns:
            True to continue streaming, False when turn is complete
        """
        if not self._vad_enabled or not self._vad_engine:
            # Fallback: basic energy-based voice detection
            audio_array = np.frombuffer(chunk.data, dtype=np.int16)
            audio_energy = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            energy_threshold = 500.0
            is_speech = audio_energy > energy_threshold

            if self._turn_detector and self._turn_detector_enabled:
                return await self._process_turn_detection(chunk, is_speech)

            return True  # Continue streaming

        try:
            # Cobra VAD processing
            audio_array = np.frombuffer(chunk.data, dtype=np.int16)
            frame_length = VAD_FRAME_LENGTH
            is_speech = False

            # Process audio in frames
            for i in range(0, len(audio_array), frame_length):
                frame = audio_array[i : i + frame_length]

                # Pad frame if needed
                if len(frame) < frame_length:
                    frame = np.pad(frame, (0, frame_length - len(frame)), "constant")

                # Get voice probability from Cobra VAD
                voice_probability = self._vad_engine.process(frame)

                if voice_probability > self._vad_threshold:
                    is_speech = True
                    self._vad_silence_counter = 0
                    self._vad_is_speaking = True
                    break
                else:
                    self._vad_silence_counter += 1

            # Process with turn detector
            if self._turn_detector and self._turn_detector_enabled:
                return await self._process_turn_detection(chunk, is_speech)

            # Update VAD speaking flag based on extended silence
            if self._vad_silence_counter >= 30:  # ~3 seconds
                self._vad_is_speaking = False

            return True  # Continue streaming

        except Exception as e:
            self._logger.error(f"Error in VAD processing: {e}")
            return True  # Continue on error

    async def _process_turn_detection(self, chunk: AudioChunk, is_speech: bool) -> bool:
        """
        Process turn detection with two-phase approach.

        Args:
            chunk: Audio chunk
            is_speech: VAD result

        Returns:
            True to continue, False when turn complete
        """
        try:
            # Skip processing if turn already completed
            if self._turn_completed_flag:
                return False  # Stop streaming
            # Add audio chunk to turn detector (manages internal state and buffers)
            state_changed, event = self._turn_detector.add_audio_chunk(
                chunk.data, is_speech, chunk.timestamp
            )

            # Handle events from turn detector
            if event == "turn_started":
                self._logger.info("Turn started - user speaking")
                self._publish_event(
                    VoiceEventType.AUDIO_STREAM_START, {"turn_in_progress": True}
                )

            elif event == "silence_detected":
                # Silence threshold reached - trigger ML analysis
                self._logger.debug("Silence detected - analyzing turn completion")
                is_complete, result = await self._turn_detector.analyze_end_of_turn(
                    force=False
                )

                if is_complete:
                    return await self._handle_turn_complete(result)

            elif event == "fallback_timeout":
                # Fallback timeout reached - force turn completion
                self._logger.info("Fallback timeout reached - forcing turn completion")
                is_complete, result = await self._turn_detector.analyze_end_of_turn(
                    force=True
                )

                if is_complete:
                    return await self._handle_turn_complete(result)

            return True  # Continue streaming

        except Exception as e:
            self._logger.error(f"Error in turn detection: {e}")
            return True  # Continue on error

    async def _handle_turn_complete(self, result: Dict[str, Any]) -> bool:
        """
        Handle turn completion.

        Args:
            result: Turn detection result

        Returns:
            False to stop streaming
        """
        try:
            prob = result.get("probability", 0.0)
            inference_time = result.get("inference_time_ms", 0.0)
            forced = result.get("forced", False)

            self._logger.info(
                f"Turn complete - prob: {prob:.3f}, "
                f"inference: {inference_time:.1f}ms, forced: {forced}"
            )

            # Set flag to prevent multiple completions
            self._turn_completed_flag = True

            # Get audio for transcription from turn detector
            audio_array = self._turn_detector.get_audio_for_transcription()

            if audio_array is not None:
                # Convert float32 numpy array to bytes (int16)
                audio_int16 = (audio_array * 32768.0).astype(np.int16)
                audio_data = audio_int16.tobytes()

                if len(audio_data) > 0:
                    # Log audio ready for transcription
                    self._publish_audio_for_transcription(audio_data)

                    # OPTIMIZATION: Publish single AUDIO_STREAM_END event immediately
                    # with stream_continues=False to signal definitive stream end
                    stats = self._turn_detector.get_statistics()
                    self._publish_event(
                        VoiceEventType.AUDIO_STREAM_END,
                        {
                            "turn_complete": True,
                            "stream_continues": False,  # Definitive stream end
                            "audio_size": len(audio_data),
                            "ready_for_transcription": True,
                            "probability": prob,
                            "inference_time_ms": inference_time,
                            "forced": forced,
                            "stats": stats,
                        },
                    )

                    # Transition to transcribing immediately after event
                    if (
                        self._state_machine.current_state
                        == AssistantState.STREAMING_AUDIO
                    ):
                        self._state_machine.transition_to(AssistantState.TRANSCRIBING)

            return False  # Stop streaming

        except Exception as e:
            self._logger.error(f"Error handling turn complete: {e}")
            return True  # Continue on error

    def _publish_audio_chunk(self, chunk: AudioChunk) -> None:
        """Publish audio chunk using optimized custom message."""
        try:
            chunk_msg = AudioChunkMsg()
            # Direct byte array assignment - no encoding needed!
            chunk_msg.data = list(chunk.data)  # Convert bytes to list of uint8
            chunk_msg.timestamp.sec = int(chunk.timestamp)
            chunk_msg.timestamp.nanosec = int((chunk.timestamp % 1) * 1e9)
            chunk_msg.sequence_id = chunk.sequence_id
            chunk_msg.format = chunk.format.name
            chunk_msg.sample_rate = chunk.sample_rate
            chunk_msg.channels = chunk.channels
            chunk_msg.sample_width = chunk.sample_width
            chunk_msg.is_final = chunk.is_final
            chunk_msg.stream_id = getattr(chunk, 'stream_id', 0)
            
            self._audio_chunk_pub.publish(chunk_msg)
            self._logger.debug(
                f"Audio chunk published to topic - seq: {chunk.sequence_id}"
            )
        except Exception as e:
            self._logger.error(f"Failed to publish audio chunk: {e}")
            raise

    def _publish_audio_for_transcription(self, audio_data: bytes) -> None:
        """Log audio ready for transcription - actual event published elsewhere."""
        # OPTIMIZATION: Don't publish duplicate AUDIO_STREAM_END event here
        # The event is already published in _handle_turn_complete() with full telemetry
        self._logger.info(f"Audio ready for transcription: {len(audio_data)} bytes")

    async def async_cleanup(self) -> None:
        """Cleanup async resources."""
        self._logger.info("Starting async cleanup")

        # Signal shutdown
        self._shutdown_event.set()

        # Cleanup ESPHome client
        if self._esphome_client:
            await self._esphome_client.cleanup()

        # Clear audio buffer
        if self._audio_buffer:
            self._audio_buffer.clear_buffer()

        # Cleanup VAD engine
        if self._vad_engine:
            try:
                self._vad_engine.delete()
                self._vad_engine = None
            except Exception as e:
                self._logger.error(f"Error cleaning up VAD engine: {e}")

        # Cleanup turn detector
        if self._turn_detector:
            try:
                self._turn_detector.cleanup()
                self._turn_detector = None
            except Exception as e:
                self._logger.error(f"Error cleaning up turn detector: {e}")

        self._logger.info("Async cleanup complete")

    def destroy_node(self) -> None:
        """Override destroy_node to ensure proper cleanup."""
        self._logger.info("Destroying voice assistant node")

        # Cleanup async components
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.async_cleanup(), self._loop)

        super().destroy_node()


def run_async_loop(node: VoiceAssistantNode) -> None:
    """Run async event loop in separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    node._loop = loop

    try:
        # Initialize async components
        loop.run_until_complete(node.async_init())

        # Run until shutdown
        loop.run_until_complete(node._shutdown_event.wait())

    except Exception as e:
        node.get_logger().error(f"Async loop error: {e}")
    finally:
        loop.close()


def main(args=None):
    """Main entry point for voice assistant core node."""
    rclpy.init(args=args)

    # Create node
    node = VoiceAssistantNode()

    # Start async thread
    async_thread = Thread(target=run_async_loop, args=(node,), daemon=True)
    async_thread.start()
    node._async_thread = async_thread

    # Create executor - optimized thread count for better performance
    executor = MultiThreadedExecutor(num_threads=2)  # Reduced from 4 to 2 threads
    executor.add_node(node)

    try:
        node.get_logger().info("Voice Assistant Core node starting...")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested")
    finally:
        # Cleanup
        node._shutdown_event.set()
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

        # Wait for async thread
        if async_thread.is_alive():
            async_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
