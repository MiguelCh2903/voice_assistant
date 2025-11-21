"""
Text-to-Speech (TTS) node for voice assistant.

This module implements a ROS2 node that handles text-to-speech synthesis using
Eleven Labs API with streaming support. It subscribes to LLM responses from the
agent and publishes audio data for playback.
"""

import json
import logging
import os
import threading
from queue import Empty, Queue
from typing import Any, Dict, Optional

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

try:
    from elevenlabs import ElevenLabs

    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("Warning: ElevenLabs SDK not available. TTS functionality will be disabled.")

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available. Local audio playback will be disabled.")


class TTSNode(Node):
    """
    Text-to-Speech processing node using Eleven Labs streaming API.

    Subscribes to LLM response sentences from agent node and provides
    real-time audio synthesis via Eleven Labs streaming.
    """

    def __init__(self):
        """Initialize the TTS node."""
        super().__init__("tts_node")

        # Setup logging
        self._setup_logging()

        # Core components
        self._elevenlabs_client: Optional[ElevenLabs] = None

        # Audio playback state
        self._audio_playback_enabled = SOUNDDEVICE_AVAILABLE
        self._audio_output_device = None
        self._audio_queue = Queue()
        self._playback_thread = None
        self._playback_active = False

        # Configuration
        self._api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self._api_key:
            self.get_logger().error("ELEVENLABS_API_KEY environment variable not set!")

        # ROS2 callback groups
        self._default_cb_group = ReentrantCallbackGroup()

        # ROS2 interface setup
        self._setup_parameters()
        self._setup_publishers()
        self._setup_subscribers()

        # State tracking
        self._synthesis_count = 0

        if ELEVENLABS_AVAILABLE and self._api_key:
            self._initialize_elevenlabs()
        else:
            self.get_logger().error(
                "TTS node disabled - missing ElevenLabs SDK or API key"
            )

        # Initialize audio playback
        if self._audio_playback_enabled:
            self._initialize_audio_playback()
            self._start_playback_thread()

        self.get_logger().info("TTS Node initialized")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        )
        self._logger = self.get_logger()

    def _setup_parameters(self) -> None:
        """Setup ROS2 parameters."""
        # Eleven Labs configuration
        self.declare_parameter("elevenlabs.voice_id", "JBFqnCBsd6RMkjVDRZzb")
        self.declare_parameter("elevenlabs.model_id", "eleven_flash_v2_5")
        self.declare_parameter("elevenlabs.output_format", "pcm_16000")

        # Audio configuration (output format for ESPHome)
        self.declare_parameter("audio.sample_rate", 16000)
        self.declare_parameter("audio.channels", 1)
        self.declare_parameter("audio.sample_width", 16)

        # Processing configuration
        self.declare_parameter("processing.stream_chunk_size", 4096)

        # Playback configuration
        self.declare_parameter("playback.enabled", True)
        self.declare_parameter("playback.device_id", -1)  # -1 for default device

        self._logger.info("TTS parameters declared")

    def _setup_publishers(self) -> None:
        """Setup ROS2 publishers."""
        # QoS profiles for audio data - use RELIABLE to ensure audio delivery
        audio_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Publishers
        self._tts_audio_pub = self.create_publisher(
            String,
            "~/tts_audio",
            audio_qos,
            callback_group=self._default_cb_group,
        )

        self._event_pub = self.create_publisher(
            String, "~/tts_event", event_qos, callback_group=self._default_cb_group
        )

        self._logger.info("TTS publishers created")

    def _setup_subscribers(self) -> None:
        """Setup ROS2 subscribers."""
        # QoS for LLM responses - RELIABLE to ensure we receive all sentences
        llm_response_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )

        # Subscribe to LLM responses from agent
        self._llm_response_sub = self.create_subscription(
            String,
            "/voice_assistant/llm_response",
            self._llm_response_callback,
            llm_response_qos,
            callback_group=self._default_cb_group,
        )

        self._logger.info("TTS subscribers created")

    def _initialize_elevenlabs(self) -> None:
        """Initialize Eleven Labs client."""
        try:
            self._elevenlabs_client = ElevenLabs(api_key=self._api_key)
            self._logger.info("ElevenLabs client initialized successfully")
        except Exception as e:
            self._logger.error(f"Failed to initialize ElevenLabs client: {e}")
            self._elevenlabs_client = None

    def _initialize_audio_playback(self) -> None:
        """Initialize audio playback device."""
        try:
            playback_enabled = self.get_parameter("playback.enabled").value
            device_id = self.get_parameter("playback.device_id").value

            if not playback_enabled:
                self._audio_playback_enabled = False
                self._logger.info("Audio playback disabled by configuration")
                return

            # Set audio device (use default if device_id is -1)
            if device_id >= 0:
                self._audio_output_device = device_id
                self._logger.info(f"Using audio output device: {device_id}")
            else:
                self._audio_output_device = None
                self._logger.info("Using default audio output device")

            # List available devices for debugging
            devices = sd.query_devices()
            self._logger.info(f"Available audio devices: {devices}")

            self._logger.info("Audio playback initialized successfully")

        except Exception as e:
            self._logger.error(f"Failed to initialize audio playback: {e}")
            self._audio_playback_enabled = False

    def _start_playback_thread(self) -> None:
        """Start background thread for audio playback."""
        self._playback_active = True
        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True
        )
        self._playback_thread.start()
        self._logger.info("Audio playback thread started")

    def _playback_worker(self) -> None:
        """Worker thread that processes audio playback queue."""
        while self._playback_active:
            try:
                # Wait for audio data with timeout
                audio_data = self._audio_queue.get(timeout=1.0)
                if audio_data is None:  # Shutdown signal
                    break

                # Get audio parameters
                sample_rate = self.get_parameter("audio.sample_rate").value
                channels = self.get_parameter("audio.channels").value

                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                if channels > 1:
                    audio_array = audio_array.reshape(-1, channels)

                self._logger.info(
                    f"Playing audio: {len(audio_data)} bytes, "
                    f"{sample_rate}Hz, {channels}ch on Bluetooth speaker"
                )

                # Play audio (blocking in this thread)
                sd.play(
                    audio_array,
                    samplerate=sample_rate,
                    device=self._audio_output_device,
                    blocking=True,
                )

                self._logger.info("Audio playback completed")

            except Empty:
                # Queue timeout - this is normal, just continue
                continue
            except Exception as e:
                if self._playback_active:
                    self._logger.error(f"Error in playback worker: {e}", exc_info=True)

    def _llm_response_callback(self, msg: String) -> None:
        """
        Handle incoming LLM response sentences.

        Args:
            msg: String message containing JSON with LLM response data
        """
        try:
            # Parse the JSON message
            data = json.loads(msg.data)
            response_text = data.get("response_text", "")
            conversation_id = data.get("conversation_id", "")
            continue_conversation = data.get("continue_conversation", False)

            if not response_text.strip():
                self._logger.debug("Received empty response text, skipping synthesis")
                return

            self._logger.info(
                f"Received LLM response (continue={continue_conversation}): "
                f"{response_text[:50]}..."
            )

            # Synthesize the sentence immediately
            self._synthesize_and_publish(response_text, conversation_id)

        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse LLM response JSON: {e}")
        except Exception as e:
            self._logger.error(f"Error in LLM response callback: {e}")

    def _synthesize_and_publish(self, text: str, conversation_id: str) -> None:
        """
        Synthesize text with Eleven Labs streaming and publish audio.

        Args:
            text: Text to synthesize
            conversation_id: Conversation ID for tracking
        """
        if not self._elevenlabs_client:
            self._logger.error("ElevenLabs client not initialized")
            return

        try:
            # Get parameters
            voice_id = self.get_parameter("elevenlabs.voice_id").value
            model_id = self.get_parameter("elevenlabs.model_id").value
            output_format = self.get_parameter("elevenlabs.output_format").value

            self._logger.info(
                f"Starting synthesis with voice={voice_id}, model={model_id}"
            )

            # Generate audio with streaming
            audio_stream = self._elevenlabs_client.text_to_speech.stream(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format=output_format,
            )

            # Accumulate audio chunks from the stream
            audio_chunks = []
            chunk_count = 0

            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    audio_chunks.append(chunk)
                    chunk_count += 1

            self._logger.info(f"Received {chunk_count} audio chunks from Eleven Labs")

            # Combine all chunks into single audio data
            full_audio = b"".join(audio_chunks)

            if not full_audio:
                self._logger.warning("No audio data received from Eleven Labs")
                return

            # Play audio locally on speaker
            if self._audio_playback_enabled:
                self._play_audio(full_audio)

            # Publish the audio message
            self._publish_tts_audio(full_audio, text, conversation_id)

            self._synthesis_count += 1
            self._logger.info(
                f"Successfully synthesized and published audio "
                f"({len(full_audio)} bytes, synthesis #{self._synthesis_count})"
            )

        except Exception as e:
            self._logger.error(f"Failed to synthesize text: {e}")
            self._publish_event("synthesis_error", {"error": str(e), "text": text})

    def _publish_tts_audio(
        self, audio_data: bytes, source_text: str, conversation_id: str
    ) -> None:
        """
        Publish TTS audio message.

        Args:
            audio_data: PCM audio bytes
            source_text: Original text that was synthesized
            conversation_id: Conversation ID for tracking
        """
        try:
            # Get audio parameters
            sample_rate = self.get_parameter("audio.sample_rate").value
            channels = self.get_parameter("audio.channels").value
            sample_width = self.get_parameter("audio.sample_width").value

            # Create message with audio data as hex string
            message_data = {
                "audio_data": audio_data.hex(),
                "format": "pcm_s16le",
                "sample_rate": sample_rate,
                "channels": channels,
                "sample_width": sample_width,
                "source_text": source_text,
                "engine": "elevenlabs",
                "conversation_id": conversation_id,
            }

            msg = String()
            msg.data = json.dumps(message_data)
            self._tts_audio_pub.publish(msg)

            self._logger.debug(
                f"Published TTS audio: {len(audio_data)} bytes, "
                f"{sample_rate}Hz, {channels}ch"
            )

        except Exception as e:
            self._logger.error(f"Failed to publish TTS audio: {e}")

    def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Publish TTS event.

        Args:
            event_type: Type of event (e.g., 'synthesis_started', 'synthesis_error')
            data: Event data dictionary
        """
        try:
            event_data = {
                "event_type": event_type,
                "timestamp": self.get_clock().now().to_msg(),
                **data,
            }

            msg = String()
            msg.data = json.dumps(event_data, default=str)
            self._event_pub.publish(msg)

            self._logger.debug(f"Published TTS event: {event_type}")

        except Exception as e:
            self._logger.error(f"Failed to publish TTS event: {e}")

    def _play_audio(self, audio_data: bytes) -> None:
        """
        Queue audio data for playback on the local speaker (Bluetooth).

        Args:
            audio_data: PCM audio bytes (16-bit, 16kHz, mono)
        """
        try:
            if not self._audio_playback_enabled:
                self._logger.debug("Audio playback disabled, skipping")
                return

            # Add audio to queue for sequential playback
            self._audio_queue.put(audio_data)
            self._logger.debug(f"Queued audio for playback: {len(audio_data)} bytes")

        except Exception as e:
            self._logger.error(f"Failed to queue audio: {e}")
            self._publish_event("playback_error", {"error": str(e)})


def main(args=None):
    """Main entry point for TTS node."""
    rclpy.init(args=args)

    try:
        node = TTSNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        try:
            node.get_logger().info("TTS Node spinning...")
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("Keyboard interrupt, shutting down...")
        finally:
            executor.shutdown()
            node.destroy_node()

    except Exception as e:
        print(f"Fatal error in TTS node: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
