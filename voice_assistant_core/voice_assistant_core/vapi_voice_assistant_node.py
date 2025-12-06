"""
Simplified ROS2 node for VAPI voice assistant integration.

This module implements a simplified voice assistant node that integrates
VAPI with ESPHome audio streaming, eliminating the need for separate
STT, LLM, and TTS nodes.
"""

import asyncio
import logging
import os
from threading import Thread
from typing import Optional

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from voice_assistant_msgs.msg import (
    AssistantState as AssistantStateMsg,
    VoiceEvent as VoiceEventMsg,
)

from .audio.types import ESPHomeDeviceInfo
from .communication import ESPHomeClientWrapper
from .vapi import VapiClient


class VapiVoiceAssistantNode(Node):
    """
    Simplified ROS2 node for VAPI voice assistant.
    
    This node manages:
    - VAPI call lifecycle
    - ESPHome audio streaming to VAPI
    - ROS2 event publishing
    """

    def __init__(self):
        """Initialize the VAPI voice assistant node."""
        super().__init__("vapi_voice_assistant")

        # Setup logging
        self._setup_logging()

        # Core components
        self._esphome_client: Optional[ESPHomeClientWrapper] = None
        self._vapi_client: Optional[VapiClient] = None

        # ROS2 callback groups
        self._default_cb_group = ReentrantCallbackGroup()

        # Async event loop management
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[Thread] = None
        self._shutdown_event = asyncio.Event()

        # ROS2 interface setup
        self._setup_parameters()
        self._setup_publishers()
        self._setup_timers()

        # State tracking
        self._call_active = False
        self._audio_streaming = False

        self.get_logger().info("VAPI Voice Assistant Node initialized")

        # Start async loop
        self._start_async_loop()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        numeric_level = getattr(logging, log_level, logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _setup_parameters(self) -> None:
        """Declare ROS2 parameters."""
        # VAPI parameters
        self.declare_parameter("vapi.api_key", "")
        self.declare_parameter("vapi.api_url", "https://api.vapi.ai")
        self.declare_parameter("vapi.assistant_id", "")
        self.declare_parameter("vapi.auto_start_call", True)

        # ESPHome device parameters
        self.declare_parameter("device.host", "")
        self.declare_parameter("device.port", 6053)
        self.declare_parameter("device.password", "")
        self.declare_parameter("device.encryption_key", "")
        self.declare_parameter("device.name", "Voice Assistant")

        # Audio parameters
        self.declare_parameter("audio.sample_rate", 16000)
        self.declare_parameter("audio.sample_width", 16)
        self.declare_parameter("audio.channels", 1)

        # Debug parameters
        self.declare_parameter("debug.log_level", "INFO")

    def _setup_publishers(self) -> None:
        """Setup ROS2 publishers."""
        self._state_publisher = self.create_publisher(
            AssistantStateMsg, "assistant_state", 1
        )

        self._event_publisher = self.create_publisher(
            VoiceEventMsg, "voice_event", 5
        )

    def _setup_timers(self) -> None:
        """Setup ROS2 timers."""
        # Status publishing timer
        self.create_timer(
            1.0, self._status_timer_callback, callback_group=self._default_cb_group
        )

    def _start_async_loop(self) -> None:
        """Start the async event loop in a separate thread."""
        self._loop = asyncio.new_event_loop()
        self._async_thread = Thread(target=self._run_async_loop, daemon=True)
        self._async_thread.start()
        self.get_logger().info("Async event loop started")

    def _run_async_loop(self) -> None:
        """Run the async event loop."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as e:
            self.get_logger().error(f"Async loop error: {e}")
        finally:
            self._loop.close()

    async def _async_main(self) -> None:
        """Main async coroutine."""
        try:
            # Initialize components
            await self._initialize_components()

            # Wait for shutdown
            await self._shutdown_event.wait()

        except Exception as e:
            self.get_logger().error(f"Error in async main: {e}")
        finally:
            await self._cleanup_components()

    async def _initialize_components(self) -> None:
        """Initialize VAPI and ESPHome components."""
        try:
            self.get_logger().info("Initializing components")

            # Initialize VAPI client
            api_key = self.get_parameter("vapi.api_key").value
            api_url = self.get_parameter("vapi.api_url").value

            if not api_key:
                raise ValueError("VAPI API key not configured")

            self._vapi_client = VapiClient(api_key=api_key, api_url=api_url)
            self._vapi_client.set_callbacks(
                on_speech_start=self._on_vapi_speech_start,
                on_speech_end=self._on_vapi_speech_end,
                on_transcript=self._on_vapi_transcript,
                on_response=self._on_vapi_response,
                on_error=self._on_vapi_error,
            )

            # Initialize ESPHome client
            device_info = ESPHomeDeviceInfo(
                host=self.get_parameter("device.host").value,
                port=self.get_parameter("device.port").value,
                password=self.get_parameter("device.password").value,
                encryption_key=self.get_parameter("device.encryption_key").value,
            )

            self._esphome_client = ESPHomeClientWrapper(device_info)
            self._esphome_client._audio_callback = self._on_esphome_audio

            # Connect to ESPHome device
            await self._esphome_client.connect()
            self.get_logger().info("Connected to ESPHome device")

            # Auto-start call if configured
            if self.get_parameter("vapi.auto_start_call").value:
                await self._start_vapi_call()

        except Exception as e:
            self.get_logger().error(f"Failed to initialize components: {e}")
            raise

    async def _start_vapi_call(self) -> None:
        """Start a VAPI call."""
        try:
            assistant_id = self.get_parameter("vapi.assistant_id").value

            if not assistant_id:
                self.get_logger().warning("VAPI assistant ID not configured")
                return

            self.get_logger().info("Starting VAPI call")
            call_id = await self._vapi_client.start_call(assistant_id=assistant_id)
            self._call_active = True
            self._audio_streaming = True

            self.get_logger().info(f"VAPI call started: {call_id}")

            # Publish event
            self._publish_event("call_started", {"call_id": call_id})

        except Exception as e:
            self.get_logger().error(f"Failed to start VAPI call: {e}")

    async def _stop_vapi_call(self) -> None:
        """Stop the current VAPI call."""
        try:
            if self._vapi_client and self._call_active:
                self.get_logger().info("Stopping VAPI call")
                await self._vapi_client.stop_call()
                self._call_active = False
                self._audio_streaming = False

                # Publish event
                self._publish_event("call_stopped", {})

        except Exception as e:
            self.get_logger().error(f"Failed to stop VAPI call: {e}")

    async def _on_esphome_audio(self, audio_data: bytes) -> None:
        """
        Handle audio data from ESPHome device.
        
        Args:
            audio_data: Raw PCM audio data
        """
        try:
            if self._audio_streaming and self._vapi_client:
                # Stream audio to VAPI
                await self._vapi_client.stream_audio(audio_data)

        except Exception as e:
            self.get_logger().error(f"Error handling ESPHome audio: {e}")

    async def _on_vapi_speech_start(self) -> None:
        """Handle VAPI speech start event."""
        self.get_logger().info("User speech started")
        self._publish_event("speech_start", {})

    async def _on_vapi_speech_end(self) -> None:
        """Handle VAPI speech end event."""
        self.get_logger().info("User speech ended")
        self._publish_event("speech_end", {})

    async def _on_vapi_transcript(self, transcript: str) -> None:
        """
        Handle VAPI transcript event.
        
        Args:
            transcript: Transcribed text
        """
        self.get_logger().info(f"Transcript: {transcript}")
        self._publish_event("transcript", {"text": transcript})

    async def _on_vapi_response(self, response: str) -> None:
        """
        Handle VAPI response event.
        
        Args:
            response: Assistant response text
        """
        self.get_logger().info(f"Response: {response}")
        self._publish_event("response", {"text": response})

    async def _on_vapi_error(self, error: str) -> None:
        """
        Handle VAPI error event.
        
        Args:
            error: Error message
        """
        self.get_logger().error(f"VAPI error: {error}")
        self._publish_event("error", {"message": error})

    def _publish_event(self, event_type: str, data: dict) -> None:
        """
        Publish a voice event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        try:
            import json
            
            msg = VoiceEventMsg()
            msg.event_type = event_type
            msg.message = f"{event_type} event occurred"
            msg.timestamp = self.get_clock().now().to_msg()
            msg.priority = VoiceEventMsg.PRIORITY_INFO
            msg.event_data = json.dumps(data) if data else ""
            self._event_publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing event: {e}")

    def _status_timer_callback(self) -> None:
        """Timer callback for publishing status."""
        try:
            msg = AssistantStateMsg()
            msg.current_state = "active" if self._call_active else "idle"
            msg.previous_state = ""
            msg.transition_time = self.get_clock().now().to_msg()
            msg.state_data = ""
            self._state_publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error in status timer: {e}")

    async def _cleanup_components(self) -> None:
        """Cleanup components on shutdown."""
        try:
            self.get_logger().info("Cleaning up components")

            # Stop VAPI call
            await self._stop_vapi_call()

            # Disconnect ESPHome
            if self._esphome_client:
                await self._esphome_client.disconnect()

        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")

    def destroy_node(self) -> None:
        """Cleanup and destroy the node."""
        self.get_logger().info("Shutting down node")

        # Signal async loop to stop
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._shutdown_event.set)

        # Wait for async thread
        if self._async_thread and self._async_thread.is_alive():
            self._async_thread.join(timeout=5.0)

        super().destroy_node()


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)

    try:
        node = VapiVoiceAssistantNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)

        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()

    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
