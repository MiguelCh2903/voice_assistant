"""
ESPHome client wrapper for voice assistant core.

This module provides a robust wrapper around aioesphomeapi with connection
management, automatic reconnection, heartbeat mechanism, and error recovery
based on Home Assistant patterns.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from aioesphomeapi import (
    APIClient,
    APIConnection,
    APIConnectionError,
    InvalidAuthAPIError,
    ReconnectLogic,
    VoiceAssistantEventType,
    VoiceAssistantFeature,
)

from ..audio.types import (
    ConnectionState,
    ESPHomeDeviceInfo,
)

# Connection constants
CONNECTION_TIMEOUT_SEC = 15.0
HEARTBEAT_INTERVAL_SEC = 30.0
INITIAL_RECONNECT_DELAY_SEC = 1.0


class ESPHomeConnectionError(Exception):
    """Raised when ESPHome connection operations fail."""

    pass


class ESPHomeClientWrapper:
    """
    Robust wrapper for ESPHome API client with connection management.

    Provides automatic reconnection, heartbeat monitoring, feature detection,
    and voice assistant event handling with error recovery.
    """

    def __init__(self, device_info: ESPHomeDeviceInfo):
        """
        Initialize ESPHome client wrapper.

        Args:
            device_info: Device connection information
        """
        self._device_info = device_info
        self._client: Optional[APIClient] = None
        self._connection: Optional[APIConnection] = None
        self._reconnect_logic: Optional[ReconnectLogic] = None

        # Connection state
        self._connection_state = ConnectionState()
        self._is_connecting = False
        self._should_reconnect = True

        # Feature detection
        self._device_features: List[str] = []
        self._voice_assistant_features: int = 0
        self._api_version: tuple = (0, 0, 0)

        # Heartbeat and monitoring
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_heartbeat = 0.0

        # Event callbacks
        self._pipeline_start_callback: Optional[Callable] = None
        self._pipeline_stop_callback: Optional[Callable] = None
        self._audio_callback: Optional[Callable] = None
        self._event_callback: Optional[Callable] = None
        self._connection_callback: Optional[Callable] = None
        self._voice_assistant_unsubscribe: Optional[Callable] = None

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

        # Logger
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self._logger.info(
            f"ESPHome client initialized for device: {device_info.host}:{device_info.port}"
        )

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to device."""
        return (
            self._connection is not None
            and self._connection.is_connected
            and self._connection_state.is_connected
        )

    @property
    def device_info(self) -> ESPHomeDeviceInfo:
        """Get device information."""
        return self._device_info

    @property
    def connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self._connection_state

    @property
    def supported_features(self) -> List[str]:
        """Get list of supported device features."""
        return self._device_features.copy()

    async def connect(self, timeout: float = CONNECTION_TIMEOUT_SEC) -> bool:
        """
        Connect to ESPHome device with enhanced error handling.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connection successful

        Raises:
            ESPHomeConnectionError: If connection fails
        """
        if self._is_connecting:
            self._logger.warning("Connection already in progress")
            return False

        if self.is_connected:
            self._logger.debug("Already connected to device")
            return True

        self._is_connecting = True
        self._connection_state.connection_attempts += 1

        try:
            self._logger.info(
                f"Connecting to ESPHome device at {self._device_info.host}:{self._device_info.port}"
            )

            # Create API client if not exists
            if self._client is None:
                self._logger.debug("Creating new API client")
                self._client = APIClient(
                    address=self._device_info.host,
                    port=self._device_info.port,
                    password=self._device_info.password,
                    noise_psk=self._device_info.encryption_key,
                    # Don't validate device name initially - let ESPHome tell us what it is
                    expected_name=None,
                )
                self._logger.debug("API client created")

            # Establish connection with timeout
            self._logger.debug(f"Attempting connection with {timeout}s timeout")
            await asyncio.wait_for(self._client.connect(), timeout=timeout)
            self._connection = self._client._connection
            self._logger.debug("Connection established")

            # Get device information
            self._logger.debug("Updating device information")
            await self._update_device_info()

            # Detect features
            self._logger.debug("Detecting device features")
            await self._detect_features()

            # Setup voice assistant subscription
            self._logger.debug("Setting up voice assistant subscription")
            await self._setup_voice_assistant()

            # Update connection state
            self._connection_state.is_connected = True
            self._connection_state.last_connection_time = time.time()
            self._connection_state.last_error = None
            self._connection_state.reconnect_delay = INITIAL_RECONNECT_DELAY_SEC

            # Start heartbeat monitoring
            self._logger.debug("Starting heartbeat monitoring")
            await self._start_heartbeat()

            self._logger.info("Successfully connected to ESPHome device")

            # Call connection callback
            if self._connection_callback:
                try:
                    self._logger.debug("Calling connection callback")
                    await self._connection_callback(True, self._device_info)
                except Exception as e:
                    self._logger.error(
                        f"Error in connection callback: {e}", exc_info=True
                    )

            return True

        except asyncio.TimeoutError:
            error_msg = f"Connection timeout after {timeout}s"
            self._connection_state.last_error = error_msg
            self._logger.error(error_msg)
            raise ESPHomeConnectionError(error_msg)

        except InvalidAuthAPIError as e:
            error_msg = f"Authentication failed: {e}"
            self._connection_state.last_error = error_msg
            self._logger.error(error_msg)
            raise ESPHomeConnectionError(error_msg)

        except APIConnectionError as e:
            error_msg = f"API connection error: {e}"
            self._connection_state.last_error = error_msg
            self._logger.error(error_msg)
            raise ESPHomeConnectionError(error_msg)

        except ESPHomeConnectionError:
            # Re-raise our own connection errors
            raise

        except Exception as e:
            error_msg = f"Unexpected connection error: {e}"
            self._connection_state.last_error = error_msg
            self._logger.error(error_msg, exc_info=True)
            raise ESPHomeConnectionError(error_msg)

        finally:
            self._is_connecting = False

    async def disconnect(self) -> None:
        """Disconnect from ESPHome device following HA cleanup patterns."""
        self._logger.info("Disconnecting from ESPHome device")

        self._should_reconnect = False

        # First, clean up voice assistant subscription (critical for device state)
        await self._cleanup_voice_assistant_subscription()

        # Stop heartbeat
        await self._stop_heartbeat()

        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._background_tasks.clear()

        # Disconnect client
        if self._client:
            try:
                await self._client.disconnect()
            except Exception as e:
                self._logger.error(f"Error during disconnect: {e}")

        # Reset state
        self._connection = None
        self._connection_state.is_connected = False

        # Call connection callback
        if self._connection_callback:
            try:
                await self._connection_callback(False, self._device_info)
            except Exception as e:
                self._logger.error(f"Error in disconnection callback: {e}")

        self._logger.info("Disconnected from ESPHome device")

    async def _cleanup_voice_assistant_subscription(self) -> None:
        """Clean up voice assistant subscription properly."""
        if self._voice_assistant_unsubscribe:
            try:
                self._logger.debug("Cleaning up voice assistant subscription")
                # Call the unsubscribe callback
                self._voice_assistant_unsubscribe()
                self._voice_assistant_unsubscribe = None
                self._logger.debug("Voice assistant subscription cleaned up")
            except Exception as e:
                self._logger.error(
                    f"Error cleaning up voice assistant subscription: {e}"
                )
                # Reset anyway
                self._voice_assistant_unsubscribe = None

    async def start_reconnect_logic(self) -> None:
        """Start automatic reconnection logic following HA patterns."""
        if self._reconnect_logic is not None:
            return

        self._should_reconnect = True

        # Don't start reconnect logic if already connected
        if self.is_connected:
            self._logger.info(
                "Already connected - reconnect logic will be started on disconnect"
            )
            return

        try:
            # Create reconnect logic
            self._reconnect_logic = ReconnectLogic(
                client=self._client,
                on_disconnect=self._on_disconnect,
                on_connect=self._on_connect,
                zeroconf_instance=None,  # Not using zeroconf
                name=self._device_info.device_name,
            )

            # Start reconnection task
            reconnect_task = asyncio.create_task(self._reconnect_logic.start())
            self._background_tasks.append(reconnect_task)

            self._logger.info("Automatic reconnection logic started")

        except Exception as e:
            self._logger.error(f"Failed to start reconnect logic: {e}")

    async def _on_connect(self) -> None:
        """Handle successful connection (called by ReconnectLogic)."""
        try:
            self._logger.info("ReconnectLogic: Connection established")

            # Update connection state
            self._connection_state.is_connected = True
            self._connection_state.last_connection_time = time.time()
            self._connection_state.last_error = None
            self._connection_state.reconnect_delay = INITIAL_RECONNECT_DELAY_SEC

            # Re-detect features after reconnection
            await self._detect_features()

            # Re-setup voice assistant subscription
            await self._setup_voice_assistant()

            # Restart heartbeat
            await self._start_heartbeat()

            # Notify connection callback
            if self._connection_callback:
                await self._connection_callback(True, self._device_info)

            self._logger.info("Reconnection completed successfully")

        except Exception as e:
            self._logger.error(f"Error in reconnection handler: {e}", exc_info=True)

    async def _on_disconnect(self) -> None:
        """Handle disconnection (called by ReconnectLogic)."""
        try:
            self._logger.warning("ReconnectLogic: Connection lost")

            # Clean up voice assistant subscription
            await self._cleanup_voice_assistant_subscription()

            # Stop heartbeat
            await self._stop_heartbeat()

            # Update connection state
            self._connection_state.is_connected = False

            # Notify connection callback
            if self._connection_callback:
                await self._connection_callback(False, self._device_info)

        except Exception as e:
            self._logger.error(f"Error in disconnection handler: {e}", exc_info=True)

    async def stop_reconnect_logic(self) -> None:
        """Stop automatic reconnection logic."""
        self._should_reconnect = False

        if self._reconnect_logic is not None:
            await self._reconnect_logic.stop()
            self._reconnect_logic = None

        self._logger.info("Automatic reconnection logic stopped")

    async def send_voice_assistant_audio(self, audio_data: bytes) -> bool:
        """
        Send audio data to device.

        Args:
            audio_data: Raw audio bytes to send

        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            raise ESPHomeConnectionError("Not connected to device")

        try:
            await self._client.send_voice_assistant_audio(audio_data)
            self._logger.debug(f"Sent audio data: {len(audio_data)} bytes")
            return True
        except Exception as e:
            self._logger.error(f"Failed to send audio data: {e}")
            return False

    async def send_voice_assistant_event(
        self, event_type: VoiceAssistantEventType, data: Dict[str, Any]
    ) -> bool:
        """
        Send voice assistant event to device.

        Args:
            event_type: Type of event to send
            data: Event data

        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            raise ESPHomeConnectionError("Not connected to device")

        try:
            await self._client.send_voice_assistant_event(event_type, data)
            self._logger.debug(f"Sent voice assistant event: {event_type}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to send voice assistant event: {e}")
            return False

    async def _update_device_info(self) -> None:
        """Update device information from connected device."""
        try:
            device_info = await self._client.device_info()

            self._device_info.device_name = device_info.name
            self._device_info.mac_address = device_info.mac_address
            self._device_info.esphome_version = device_info.esphome_version
            # Use compilation_time or project_version as fallback for api_version
            self._api_version = getattr(device_info, "project_version", "0.0.0")

            self._logger.info(
                f"Device info updated - Name: {device_info.name}, "
                f"Version: {device_info.esphome_version}, "
                f"Project: {getattr(device_info, 'project_version', 'unknown')}"
            )

        except Exception as e:
            self._logger.error(f"Failed to update device info: {e}")

    async def _detect_features(self) -> None:
        """Detect supported device features using Home Assistant patterns."""
        try:
            # Get device info first
            device_info = await self._client.device_info()

            # Use the same method as Home Assistant for feature detection
            if hasattr(device_info, "voice_assistant_feature_flags"):
                self._voice_assistant_features = (
                    device_info.voice_assistant_feature_flags
                )
                self._logger.debug(
                    f"Device reports voice assistant features: {self._voice_assistant_features}"
                )
            else:
                # Fallback: try to detect if device supports voice assistant at all
                self._logger.debug(
                    "No voice assistant feature flags in device_info, checking entities"
                )
                entities = await self._client.list_entities_services()

                # Look for any voice assistant related entities
                found_voice_features = False
                for entity in entities:
                    entity_type = type(entity).__name__
                    if any(
                        keyword in entity_type.lower()
                        for keyword in ["voice", "assist", "microphone"]
                    ):
                        self._logger.debug(f"Found voice-related entity: {entity_type}")
                        found_voice_features = True
                        break

                if found_voice_features:
                    # Assume basic voice assistant support
                    self._voice_assistant_features = getattr(
                        VoiceAssistantFeature, "API_AUDIO", 1
                    ) | getattr(VoiceAssistantFeature, "SPEAKER", 2)
                    self._logger.debug("Assuming basic voice assistant features")
                else:
                    self._logger.warning("No voice assistant features detected")
                    self._voice_assistant_features = 0

            # Map features to our internal list following HA patterns
            self._device_features = []

            if self._voice_assistant_features:
                self._logger.info(
                    f"Voice assistant features detected: 0x{self._voice_assistant_features:x}"
                )

                # Check for specific features following HA naming
                feature_map = {
                    "API_AUDIO": "api_audio",
                    "SPEAKER": "speaker",
                    "ANNOUNCE": "announce",
                    "TIMERS": "timers",
                }

                for feature_name, internal_name in feature_map.items():
                    if hasattr(VoiceAssistantFeature, feature_name):
                        feature_flag = getattr(VoiceAssistantFeature, feature_name)
                        if self._voice_assistant_features & feature_flag:
                            self._device_features.append(internal_name)
                            self._logger.debug(f"Device supports {internal_name}")

                # Always assume microphone support if voice assistant features are present
                self._device_features.append("microphone")
                self._logger.debug("Assuming microphone support")
            else:
                self._logger.warning(
                    "No voice assistant features detected - device may not support voice assistant"
                )

            self._device_info.supported_features = self._device_features

            self._logger.info(f"Detected features: {self._device_features}")

        except Exception as e:
            self._logger.error(f"Failed to detect features: {e}", exc_info=True)

    async def _setup_voice_assistant(self) -> None:
        """Setup voice assistant event subscription following HA patterns."""
        try:
            self._logger.info("Setting up voice assistant subscription")

            # Ensure we have a valid client and connection
            if not self._client:
                raise ESPHomeConnectionError(
                    "Client not available for voice assistant subscription"
                )

            if not self._connection or not self._connection.is_connected:
                raise ESPHomeConnectionError(
                    "No active connection for voice assistant subscription"
                )

            # Check device features before subscribing (following HA pattern)
            device_info = await self._client.device_info()
            feature_flags = getattr(device_info, "voice_assistant_feature_flags", 0)

            if feature_flags == 0:
                self._logger.warning("Device reports no voice assistant features")
                # Continue anyway - some devices might not report features correctly

            # Subscribe to voice assistant events following HA pattern exactly
            self._logger.debug("Subscribing to voice assistant events...")
            try:
                # Use the exact same pattern as Home Assistant
                if feature_flags & getattr(VoiceAssistantFeature, "API_AUDIO", 1):
                    # TCP audio mode
                    self._logger.debug("Setting up TCP audio subscription")
                    unsubscribe_callback = self._client.subscribe_voice_assistant(
                        handle_start=self._handle_pipeline_start,
                        handle_stop=self._handle_pipeline_stop,
                        handle_audio=self._handle_audio,
                        handle_announcement_finished=self._handle_announcement_finished,
                    )
                else:
                    # UDP audio mode (legacy)
                    self._logger.debug("Setting up UDP audio subscription")
                    unsubscribe_callback = self._client.subscribe_voice_assistant(
                        handle_start=self._handle_pipeline_start,
                        handle_stop=self._handle_pipeline_stop,
                        handle_announcement_finished=self._handle_announcement_finished,
                    )

                # Store unsubscribe callback for cleanup - this is critical
                self._voice_assistant_unsubscribe = unsubscribe_callback
                self._logger.info(
                    "Voice assistant subscription established successfully"
                )

                # Send initial configuration to device (following HA pattern)
                await self._send_initial_configuration()

            except AttributeError as e:
                self._logger.error(
                    f"Client does not support voice assistant subscription: {e}"
                )
                raise ESPHomeConnectionError(f"Voice assistant not supported: {e}")
            except TypeError as e:
                self._logger.error(
                    f"Invalid parameters for voice assistant subscription: {e}"
                )
                raise ESPHomeConnectionError(f"Subscription parameter error: {e}")

        except ESPHomeConnectionError:
            # Re-raise connection errors
            raise
        except Exception as e:
            self._logger.error(
                f"Unexpected error during voice assistant setup: {e}", exc_info=True
            )
            raise ESPHomeConnectionError(f"Failed to setup voice assistant: {e}")

    async def _send_initial_configuration(self) -> None:
        """Send initial configuration to device following HA pattern."""
        try:
            # This helps the device know we're ready to receive voice assistant events
            if hasattr(self._client, "get_voice_assistant_configuration"):
                try:
                    config = await asyncio.wait_for(
                        self._client.get_voice_assistant_configuration(timeout=5.0),
                        timeout=5.0,
                    )
                    self._logger.debug(f"Device voice assistant config: {config}")
                except asyncio.TimeoutError:
                    self._logger.debug("Device configuration timeout - using defaults")
                except Exception as e:
                    self._logger.debug(f"Failed to get device config: {e}")

        except Exception as e:
            self._logger.debug(f"Initial configuration failed: {e}")
            # Not critical - continue anyway

    async def _handle_announcement_finished(self, success: bool) -> None:
        """Handle announcement finished event."""
        self._logger.debug(f"Announcement finished - Success: {success}")
        # For now, just log the event - can be extended later

    async def _handle_pipeline_start(
        self, conversation_id: str, flags: int, audio_settings, wake_word_phrase: str
    ) -> int:
        """
        Handle pipeline start event from device with enhanced logging.

        Args:
            conversation_id: Unique conversation identifier
            flags: Pipeline flags
            audio_settings: Audio configuration
            wake_word_phrase: Detected wake word phrase (can be None)

        Returns:
            Port number for audio streaming (0 for API mode)
        """
        self._logger.info(
            f"Pipeline started - Conversation: {conversation_id}, "
            f"Wake word: '{wake_word_phrase}', Flags: {flags}"
        )

        # Log audio settings for debugging
        if audio_settings:
            # Log the audio settings object structure for debugging
            self._logger.debug(f"Audio settings type: {type(audio_settings)}")
            self._logger.debug(f"Audio settings attributes: {dir(audio_settings)}")

            # Try to extract common audio setting attributes
            settings_info = {}
            for attr in [
                "sample_rate",
                "sample_frequency",
                "bits_per_sample",
                "sample_width",
                "channels",
                "num_channels",
                "format",
            ]:
                if hasattr(audio_settings, attr):
                    settings_info[attr] = getattr(audio_settings, attr)

            if settings_info:
                self._logger.debug(f"Audio settings values: {settings_info}")
            else:
                # Fallback: try to convert to string
                self._logger.debug(f"Audio settings (str): {str(audio_settings)}")
        else:
            self._logger.debug("No audio settings provided")

        # Call pipeline start callback with error handling
        if self._pipeline_start_callback:
            try:
                self._logger.debug("Calling pipeline start callback")
                result = await self._pipeline_start_callback(
                    conversation_id, flags, audio_settings, wake_word_phrase
                )
                self._logger.debug(f"Pipeline start callback returned: {result}")
                return result if result is not None else 0
            except Exception as e:
                self._logger.error(
                    f"Error in pipeline start callback: {e}", exc_info=True
                )
                # Return 0 to continue with API audio mode despite callback error
                return 0
        else:
            self._logger.debug("No pipeline start callback registered")

        return 0  # Use API audio mode by default

    async def _handle_pipeline_stop(self, abort: bool) -> None:
        """
        Handle pipeline stop event from device with enhanced logging.

        Args:
            abort: True if pipeline was aborted
        """
        self._logger.info(f"Pipeline stopped - Abort: {abort}")

        # Call pipeline stop callback with error handling
        if self._pipeline_stop_callback:
            try:
                self._logger.debug("Calling pipeline stop callback")
                await self._pipeline_stop_callback(abort)
                self._logger.debug("Pipeline stop callback completed")
            except Exception as e:
                self._logger.error(
                    f"Error in pipeline stop callback: {e}", exc_info=True
                )
        else:
            self._logger.debug("No pipeline stop callback registered")

    async def _handle_audio(self, data: bytes) -> None:
        """
        Handle audio data from device with enhanced logging.

        Args:
            data: Raw audio data bytes
        """
        self._logger.debug(f"Received audio data: {len(data)} bytes")

        # Call audio callback with error handling
        if self._audio_callback:
            try:
                await self._audio_callback(data)
            except Exception as e:
                self._logger.error(f"Error in audio callback: {e}", exc_info=True)
        else:
            self._logger.warning(
                f"Received audio data but no callback registered - {len(data)} bytes lost"
            )

    async def _start_heartbeat(self) -> None:
        """Start heartbeat monitoring."""
        if self._heartbeat_task is not None:
            return

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._background_tasks.append(self._heartbeat_task)

        self._logger.debug("Heartbeat monitoring started")

    async def _stop_heartbeat(self) -> None:
        """Stop heartbeat monitoring."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        self._logger.debug("Heartbeat monitoring stopped")

    async def _heartbeat_loop(self) -> None:
        """Simplified heartbeat monitoring following HA patterns."""
        while self._should_reconnect and self.is_connected:
            try:
                # Simply check if connection is still active
                # Don't make API calls unless necessary - just check connection state
                if self._connection and self._connection.is_connected:
                    self._last_heartbeat = time.time()
                    self._logger.debug("Heartbeat check: Connection active")
                else:
                    self._logger.warning("Heartbeat check: Connection inactive")
                    break

                await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)

            except asyncio.CancelledError:
                self._logger.debug("Heartbeat cancelled")
                break
            except Exception as e:
                self._logger.warning(f"Heartbeat error: {e}")
                # Don't break immediately - let reconnect logic handle it
                await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
                break

    # Remove duplicate _on_connect and _on_disconnect methods as they are now defined above

    def set_pipeline_start_callback(self, callback: Callable) -> None:
        """Set callback for pipeline start events."""
        self._pipeline_start_callback = callback

    def set_pipeline_stop_callback(self, callback: Callable) -> None:
        """Set callback for pipeline stop events."""
        self._pipeline_stop_callback = callback

    def set_audio_callback(self, callback: Callable) -> None:
        """Set callback for audio data events."""
        self._audio_callback = callback

    def set_event_callback(self, callback: Callable) -> None:
        """Set callback for general events."""
        self._event_callback = callback

    def set_connection_callback(self, callback: Callable) -> None:
        """Set callback for connection state changes."""
        self._connection_callback = callback

    async def cleanup(self) -> None:
        """Cleanup resources and connections with robust error handling."""
        self._logger.info("Cleaning up ESPHome client")

        # Set flag to prevent reconnections during cleanup
        self._should_reconnect = False

        # Clean up voice assistant subscription
        await self._cleanup_voice_assistant_subscription()

        # Stop heartbeat
        try:
            await self._stop_heartbeat()
        except Exception as e:
            self._logger.error(f"Error stopping heartbeat: {e}")

        # Stop reconnect logic
        try:
            await self.stop_reconnect_logic()
        except Exception as e:
            self._logger.error(f"Error stopping reconnect logic: {e}")

        # Cancel all background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._background_tasks.clear()

        # Disconnect client
        if self._client:
            try:
                await self._client.disconnect()
            except Exception as e:
                self._logger.error(f"Error disconnecting client: {e}")

        self._logger.info("ESPHome client cleanup completed")

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive client status with subscription info.

        Returns:
            Dictionary with client status and statistics
        """
        return {
            "is_connected": self.is_connected,
            "device_info": {
                "host": self._device_info.host,
                "port": self._device_info.port,
                "name": self._device_info.device_name,
                "mac_address": self._device_info.mac_address,
                "esphome_version": self._device_info.esphome_version,
                "supported_features": self._device_info.supported_features,
            },
            "connection_state": {
                "connection_attempts": self._connection_state.connection_attempts,
                "last_connection_time": self._connection_state.last_connection_time,
                "last_error": self._connection_state.last_error,
                "reconnect_delay": self._connection_state.reconnect_delay,
            },
            "heartbeat": {
                "last_heartbeat": self._last_heartbeat,
                "interval": HEARTBEAT_INTERVAL_SEC,
            },
            "voice_assistant": {
                "subscription_active": self._voice_assistant_unsubscribe is not None,
                "features": self._voice_assistant_features,
                "callbacks_registered": {
                    "pipeline_start": self._pipeline_start_callback is not None,
                    "pipeline_stop": self._pipeline_stop_callback is not None,
                    "audio": self._audio_callback is not None,
                    "connection": self._connection_callback is not None,
                },
            },
            "api_version": self._api_version,
            "background_tasks": len(self._background_tasks),
            "should_reconnect": self._should_reconnect,
            "is_connecting": self._is_connecting,
        }

    def validate_subscription_state(self) -> Dict[str, Any]:
        """
        Validate the current state of voice assistant subscription.

        Returns:
            Dictionary with validation results and recommendations
        """
        issues = []
        recommendations = []

        # Check basic connection
        if not self.is_connected:
            issues.append("Device not connected")
            recommendations.append("Ensure device is powered on and network accessible")

        # Check voice assistant subscription
        if self._voice_assistant_unsubscribe is None:
            issues.append("Voice assistant subscription not active")
            recommendations.append("Check if device supports voice assistant feature")

        # Check callbacks
        if not self._pipeline_start_callback:
            issues.append("No pipeline start callback registered")
            recommendations.append("Register pipeline start callback before connecting")

        if not self._audio_callback:
            issues.append("No audio callback registered")
            recommendations.append("Register audio callback to receive audio data")

        # Check features
        if not self._device_features:
            issues.append("No device features detected")
            recommendations.append("Verify device firmware supports voice assistant")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "timestamp": time.time(),
        }
