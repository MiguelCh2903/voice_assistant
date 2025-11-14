"""
Speech-to-Text (STT) node for voice assistant.

This module implements a ROS2 node that handles speech-to-text processing using
Deepgram's Nova-3 model via WebSocket streaming. It subscribes to audio chunks
from the voice assistant core and publishes transcription results.
"""

import asyncio
import json
import logging
import os
import time
from threading import Thread
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

try:
    from deepgram import AsyncDeepgramClient
    from deepgram.core.events import EventType
    from deepgram.extensions.types.sockets import ListenV1SocketClientResponse

    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False

    # Define placeholder types when Deepgram is not available
    class AsyncDeepgramClient:
        pass

    class EventType:
        pass

    class ListenV1SocketClientResponse:
        pass

    print(
        "Warning: Deepgram SDK v5+ not available. STT functionality will be disabled."
    )


class STTNode(Node):
    """
    Speech-to-Text processing node using Deepgram Nova-3 model.

    Subscribes to audio chunks from voice assistant core and provides
    real-time transcription via WebSocket connection to Deepgram API.
    """

    def __init__(self):
        """Initialize the STT node."""
        super().__init__("stt_node")

        # Setup logging
        self._setup_logging()

        # Core components
        self._deepgram_client: Optional[AsyncDeepgramClient] = None
        self._websocket_connection = None
        self._connection_task = None

        # Audio processing state
        self._is_streaming = False
        self._audio_buffer: List[bytes] = []
        self._partial_transcripts: List[str] = []
        self._final_transcript = ""

        # Configuration
        self._api_key = os.getenv("DEEPGRAM_API_KEY")
        if not self._api_key:
            self.get_logger().error("DEEPGRAM_API_KEY environment variable not set!")

        # ROS2 callback groups
        self._default_cb_group = ReentrantCallbackGroup()
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()

        # Async event loop management
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[Thread] = None
        self._shutdown_event = asyncio.Event()

        # ROS2 interface setup
        self._setup_parameters()
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_timers()

        # State tracking
        self._last_audio_timestamp = 0.0
        self._transcription_start_time = 0.0
        self._is_turn_complete = False

        if DEEPGRAM_AVAILABLE and self._api_key:
            self._initialize_deepgram()
        else:
            self.get_logger().error(
                "STT node disabled - missing Deepgram SDK or API key"
            )

        self.get_logger().info("STT Node initialized")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        )
        self._logger = self.get_logger()

    def _setup_parameters(self) -> None:
        """Setup ROS2 parameters."""
        # Deepgram configuration - Nova-3 supports Spanish natively
        self.declare_parameter("deepgram.model", "nova-3")
        self.declare_parameter("deepgram.language", "es-419")  # Spanish by default
        self.declare_parameter("deepgram.smart_format", True)
        self.declare_parameter("deepgram.punctuate", True)
        self.declare_parameter("deepgram.interim_results", True)

        # Audio configuration
        self.declare_parameter("audio.sample_rate", 16000)
        self.declare_parameter("audio.channels", 1)
        self.declare_parameter("audio.encoding", "linear16")

        # Processing configuration
        self.declare_parameter("processing.timeout_seconds", 30.0)
        self.declare_parameter("processing.silence_timeout_seconds", 3.0)
        self.declare_parameter("processing.min_confidence", 0.5)

        self._logger.info("STT parameters declared")

    def _setup_publishers(self) -> None:
        """Setup ROS2 publishers."""
        # QoS profiles for different message types
        transcription_qos = QoSProfile(
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
        self._transcription_pub = self.create_publisher(
            String,
            "~/transcription_result",
            transcription_qos,
            callback_group=self._default_cb_group,
        )

        self._event_pub = self.create_publisher(
            String, "~/stt_event", event_qos, callback_group=self._default_cb_group
        )

        self._logger.info("STT publishers created")

    def _setup_subscribers(self) -> None:
        """Setup ROS2 subscribers."""
        # QoS for audio chunks - best effort for real-time performance
        audio_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,  # Buffer some chunks to handle timing variations
        )

        # Use RELIABLE for voice events to match publisher
        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribe to audio chunks from voice assistant core
        self._audio_chunk_sub = self.create_subscription(
            String,
            "/voice_assistant/voice_assistant_core/audio_chunk",
            self._audio_chunk_callback,
            audio_qos,
            callback_group=self._default_cb_group,
        )

        # Subscribe to voice events to detect turn completion
        self._voice_event_sub = self.create_subscription(
            String,
            "/voice_assistant/voice_assistant_core/voice_event",
            self._voice_event_callback,
            event_qos,
            callback_group=self._default_cb_group,
        )

        self._logger.info("STT subscribers created")

    def _setup_timers(self) -> None:
        """Setup ROS2 timers."""
        # Heartbeat timer for connection monitoring
        self._heartbeat_timer = self.create_timer(
            10.0,  # Check every 10 seconds
            self._heartbeat_callback,
            callback_group=self._timer_cb_group,
        )

        # Timeout timer for processing
        self._timeout_timer = self.create_timer(
            1.0,  # Check every second
            self._timeout_callback,
            callback_group=self._timer_cb_group,
        )

        self._logger.info("STT timers created")

    def _initialize_deepgram(self) -> None:
        """Initialize Deepgram client and configuration."""
        try:
            # Create Deepgram client with API key
            # In Deepgram v5, the client uses the API key from environment by default
            # or we can pass it directly
            self._deepgram_client = AsyncDeepgramClient(api_key=self._api_key)

            self._logger.info("Deepgram client initialized successfully")

        except Exception as e:
            self._logger.error(f"Failed to initialize Deepgram client: {e}")
            self._deepgram_client = None

    async def async_init(self) -> None:
        """Initialize async components."""
        try:
            self._logger.info("Initializing STT async components")

            # Any additional async initialization can go here

            self._logger.info("STT async initialization complete")

        except Exception as e:
            self._logger.error(f"STT async initialization failed: {e}")
            raise

    # ROS2 Callbacks
    def _audio_chunk_callback(self, msg: String) -> None:
        """Handle incoming audio chunks from voice assistant core."""
        try:
            self._logger.debug("Received audio chunk callback")
            # Get current event loop for async processing
            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._process_audio_chunk(msg), self._loop
                )
        except Exception as e:
            self._logger.error(f"Error in audio chunk callback: {e}")

    async def _process_audio_chunk(self, msg: String) -> None:
        """Process incoming audio chunk asynchronously."""
        try:
            # Parse audio chunk data
            chunk_data = json.loads(msg.data)
            audio_bytes = bytes.fromhex(chunk_data["data"])
            timestamp = chunk_data["timestamp"]

            # Update last audio timestamp
            self._last_audio_timestamp = timestamp

            # If this is the start of a new stream, reset state
            if not self._is_streaming:
                await self._start_streaming_session()

            # Add audio to buffer and send to Deepgram
            self._audio_buffer.append(audio_bytes)

            if self._websocket_connection:
                await self._send_audio_to_deepgram(audio_bytes)

            self._logger.info(f"Processed audio chunk: {len(audio_bytes)} bytes")

        except json.JSONDecodeError as e:
            self._logger.error(f"Invalid audio chunk JSON: {e}")
        except Exception as e:
            self._logger.error(f"Error processing audio chunk: {e}")

    def _voice_event_callback(self, msg: String) -> None:
        """Handle voice events from voice assistant core."""
        try:
            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._process_voice_event(msg), self._loop
                )
        except Exception as e:
            self._logger.error(f"Error in voice event callback: {e}")

    async def _process_voice_event(self, msg: String) -> None:
        """Process voice events asynchronously."""
        try:
            event_data = json.loads(msg.data)
            event_type = event_data.get("event_type", "")
            data = event_data.get("data", {})

            if event_type == "AUDIO_STREAM_START":
                self._logger.info("Received audio stream start event")
                # Stream start is handled in _process_audio_chunk

            elif event_type == "AUDIO_STREAM_END":
                # Only react to confirmed turn completions, ignoring intermediate boundaries
                turn_complete = data.get("turn_complete", False)
                stream_continues = data.get("stream_continues", False)

                if turn_complete:
                    if stream_continues:
                        # Boundary detected; allow further speech before concluding
                        # Defer transcription finalization until the stream truly ends
                        self._logger.info(
                            "Turn boundary detected (stream continues) - "
                            "buffering for potential user continuation"
                        )
                    else:
                        # Stream has conclusively ended; finalize transcription
                        self._logger.info("Stream ended - finalizing transcription")
                        await self._finalize_transcription()

        except json.JSONDecodeError as e:
            self._logger.error(f"Invalid voice event JSON: {e}")
        except Exception as e:
            self._logger.error(f"Error processing voice event: {e}")

    def _heartbeat_callback(self) -> None:
        """Periodic heartbeat for monitoring."""
        try:
            # Log statistics periodically
            self._logger.debug(
                f"STT heartbeat - streaming: {self._is_streaming}, "
                f"buffer size: {len(self._audio_buffer)}"
            )
        except Exception as e:
            self._logger.error(f"Error in heartbeat callback: {e}")

    def _timeout_callback(self) -> None:
        """Check for processing timeouts."""
        try:
            if self._is_streaming:
                current_time = time.time()
                timeout_seconds = self.get_parameter("processing.timeout_seconds").value

                # Check if we've been streaming too long without completion
                if (current_time - self._transcription_start_time) > timeout_seconds:
                    self._logger.warning(
                        "STT processing timeout - finalizing transcription"
                    )
                    if self._loop and not self._loop.is_closed():
                        asyncio.run_coroutine_threadsafe(
                            self._finalize_transcription(), self._loop
                        )

        except Exception as e:
            self._logger.error(f"Error in timeout callback: {e}")

    # Deepgram WebSocket handling
    async def _start_streaming_session(self) -> None:
        """Start a new Deepgram streaming session."""
        try:
            if not self._deepgram_client:
                self._logger.error(
                    "Cannot start streaming - Deepgram client not available"
                )
                return

            self._logger.info("Starting Deepgram streaming session")

            # Reset state
            self._is_streaming = True
            self._audio_buffer.clear()
            self._partial_transcripts.clear()
            self._final_transcript = ""
            self._transcription_start_time = time.time()
            self._is_turn_complete = False

            # Get configuration parameters
            model = self.get_parameter("deepgram.model").value
            language = self.get_parameter("deepgram.language").value
            sample_rate = self.get_parameter("audio.sample_rate").value

            # Create WebSocket connection as a background task
            self._connection_task = asyncio.create_task(
                self._setup_deepgram_connection(model, language, sample_rate)
            )

            self._logger.info("Deepgram streaming session started successfully")

        except Exception as e:
            self._logger.error(f"Failed to start Deepgram streaming session: {e}")
            self._is_streaming = False

    async def _setup_deepgram_connection(
        self, model: str, language: str, sample_rate: int
    ):
        """Setup Deepgram connection using async context manager."""
        # Create connection using async with
        async with self._deepgram_client.listen.v1.connect(
            model=model,
            language=language,
            sample_rate=sample_rate,
            encoding="linear16",
        ) as connection:
            # Store the connection reference
            self._websocket_connection = connection

            # Set up event handlers
            connection.on(EventType.OPEN, self._on_deepgram_open)
            connection.on(EventType.MESSAGE, self._on_deepgram_message)
            connection.on(EventType.CLOSE, self._on_deepgram_close)
            connection.on(EventType.ERROR, self._on_deepgram_error)

            # Start listening - this will run indefinitely until connection closes
            await connection.start_listening()

    async def _send_audio_to_deepgram(self, audio_data: bytes) -> None:
        """Send audio data to Deepgram WebSocket."""
        try:
            if self._websocket_connection and self._is_streaming:
                # Use V1 API for Nova-3 model - await the async send_media
                await self._websocket_connection.send_media(audio_data)

        except Exception as e:
            self._logger.error(f"Error sending audio to Deepgram: {e}")

    def _on_deepgram_open(self, open_result) -> None:
        """Handle Deepgram WebSocket open event."""
        self._logger.info("Deepgram WebSocket connection opened")

    def _on_deepgram_message(self, message: ListenV1SocketClientResponse) -> None:
        """Handle Deepgram transcription messages."""
        try:
            if hasattr(message, "channel") and hasattr(message.channel, "alternatives"):
                transcript = message.channel.alternatives[0].transcript

                if len(transcript) > 0:
                    is_final = getattr(message, "is_final", False)

                    if is_final:
                        # Final transcript
                        self._partial_transcripts.append(transcript)
                        self._logger.info(f"Final transcript segment: {transcript}")
                    else:
                        # Interim transcript
                        self._logger.debug(f"Interim transcript: {transcript}")

        except Exception as e:
            self._logger.error(f"Error processing Deepgram message: {e}")

    def _on_deepgram_close(self, close_result) -> None:
        """Handle Deepgram WebSocket close event."""
        self._logger.info("Deepgram WebSocket connection closed")

    def _on_deepgram_error(self, error) -> None:
        """Handle Deepgram WebSocket error event."""
        self._logger.error(f"Deepgram WebSocket error: {error}")

    async def _finalize_transcription(self) -> None:
        """Finalize and publish the transcription result."""
        try:
            self._logger.info("Finalizing transcription")

            # Maintain streaming state so the session stays ready for upcoming audio
            # self._is_streaming = False

            # Preserve the WebSocket connection for reuse in the next turn
            if self._websocket_connection:
                try:
                    # Continuous Deepgram streaming does not require an explicit finish signal
                    # Keep-alive traffic ensures the socket remains open for future audio
                    # await self._websocket_connection.finish()
                    self._logger.info(
                        "WebSocket connection kept alive for continuous streaming"
                    )
                except Exception as e:
                    self._logger.warning(f"Error in WebSocket handling: {e}")

            # Combine all partial transcripts
            complete_transcript = " ".join(self._partial_transcripts).strip()

            if complete_transcript:
                # Calculate processing time
                processing_time = time.time() - self._transcription_start_time

                # Create transcription result message
                result_data = {
                    "text": complete_transcript,
                    "confidence": 0.9,  # Deepgram doesn't provide confidence in Nova model
                    "language": self.get_parameter("deepgram.language").value,
                    "processing_time": processing_time,
                    "audio_metadata": {
                        "sample_rate": self.get_parameter("audio.sample_rate").value,
                        "channels": 1,
                        "encoding": "linear16",
                        "duration_seconds": processing_time,
                    },
                }

                # Publish transcription result
                result_msg = String()
                result_msg.data = json.dumps(result_data)
                self._transcription_pub.publish(result_msg)

                self._logger.info(
                    f"Published complete transcription: '{complete_transcript}' "
                    f"(processing time: {processing_time:.2f}s)"
                )

                # Publish completion event
                self._publish_event(
                    "TRANSCRIPTION_COMPLETE",
                    {
                        "transcript_length": len(complete_transcript),
                        "processing_time": processing_time,
                        "audio_chunks_processed": len(self._audio_buffer),
                    },
                )
            else:
                self._logger.warning("No transcription generated")

                # Publish empty transcription event
                self._publish_event(
                    "TRANSCRIPTION_EMPTY",
                    {
                        "processing_time": time.time() - self._transcription_start_time,
                        "audio_chunks_processed": len(self._audio_buffer),
                    },
                )

            # Clear transcript buffers only; retain the audio buffer and socket
            self._partial_transcripts.clear()
            self._final_transcript = ""
            # Reset the transcription timer for the next turn
            self._transcription_start_time = time.time()

        except Exception as e:
            self._logger.error(f"Error finalizing transcription: {e}")

    def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish STT event."""
        try:
            event_data = {
                "event_type": event_type,
                "timestamp": time.time(),
                "data": data,
            }

            event_msg = String()
            event_msg.data = json.dumps(event_data)
            self._event_pub.publish(event_msg)

        except Exception as e:
            self._logger.error(f"Error publishing event: {e}")

    # Cleanup
    async def async_cleanup(self) -> None:
        """Cleanup async resources."""
        self._logger.info("Starting STT async cleanup")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop streaming if active
        if self._is_streaming and self._connection_task:
            try:
                # Cancel the connection task
                self._connection_task.cancel()
                try:
                    await self._connection_task
                except asyncio.CancelledError:
                    pass
            except Exception as e:
                self._logger.error(f"Error stopping WebSocket connection: {e}")

        # Cleanup Deepgram client
        self._deepgram_client = None
        self._websocket_connection = None
        self._connection_task = None

        self._logger.info("STT async cleanup complete")

    def destroy_node(self) -> None:
        """Override destroy_node to ensure proper cleanup."""
        self._logger.info("Destroying STT node")

        # Cleanup async components
        if self._loop and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.async_cleanup(), self._loop)

        super().destroy_node()


def run_async_loop(node: STTNode) -> None:
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
        node.get_logger().error(f"STT async loop error: {e}")
    finally:
        loop.close()


def main(args=None):
    """Main entry point for STT node."""
    rclpy.init(args=args)

    # Create STT node
    node = STTNode()

    # Start async thread
    async_thread = Thread(target=run_async_loop, args=(node,), daemon=True)
    async_thread.start()
    node._async_thread = async_thread

    # Create executor
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        node.get_logger().info("STT Node starting...")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("STT shutdown requested")
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
