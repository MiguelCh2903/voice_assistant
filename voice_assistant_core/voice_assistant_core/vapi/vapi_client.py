"""
VAPI client wrapper for ROS2 integration.

This module provides a wrapper around the VAPI Python SDK that integrates
with ESPHome audio streaming and ROS2 event handling.
"""

import asyncio
import logging
from typing import Callable, Optional

from vapi_python import Vapi


class VapiClient:
    """
    Wrapper for VAPI SDK with ESPHome audio streaming support.
    
    This client manages the VAPI call lifecycle and coordinates audio streaming
    from ESPHome devices without requiring wake word activation.
    """

    def __init__(self, api_key: str, api_url: str = "https://api.vapi.ai"):
        """
        Initialize VAPI client.
        
        Args:
            api_key: VAPI API key
            api_url: VAPI API URL (default: https://api.vapi.ai)
        """
        self._api_key = api_key
        self._api_url = api_url
        self._vapi: Optional[Vapi] = None
        
        # Call state
        self._call_active = False
        self._call_id: Optional[str] = None
        
        # Audio streaming
        self._audio_queue = asyncio.Queue()
        self._stream_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
        self._on_transcript: Optional[Callable] = None
        self._on_response: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        
        # Logger
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        self._logger.info("VAPI client initialized")

    @property
    def is_call_active(self) -> bool:
        """Check if call is currently active."""
        return self._call_active

    @property
    def call_id(self) -> Optional[str]:
        """Get current call ID."""
        return self._call_id

    def set_callbacks(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        on_transcript: Optional[Callable] = None,
        on_response: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """
        Set event callbacks for VAPI events.
        
        Args:
            on_speech_start: Called when user speech starts
            on_speech_end: Called when user speech ends
            on_transcript: Called when transcript is received
            on_response: Called when assistant response is received
            on_error: Called when an error occurs
        """
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_transcript = on_transcript
        self._on_response = on_response
        self._on_error = on_error
        
        self._logger.debug("Event callbacks configured")

    async def start_call(
        self,
        assistant_id: Optional[str] = None,
        assistant: Optional[dict] = None,
        assistant_overrides: Optional[dict] = None,
    ) -> str:
        """
        Start a new VAPI call.
        
        Args:
            assistant_id: VAPI assistant ID
            assistant: Assistant configuration dict
            assistant_overrides: Override parameters for the assistant
            
        Returns:
            Call ID
            
        Raises:
            Exception: If call cannot be started
        """
        if self._call_active:
            self._logger.warning("Call already active")
            return self._call_id

        try:
            self._logger.info("Starting VAPI call")
            
            # Create VAPI instance
            self._vapi = Vapi(api_key=self._api_key, api_url=self._api_url)
            
            # Start call in a separate thread (VAPI SDK uses blocking operations)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._start_call_blocking,
                assistant_id,
                assistant,
                assistant_overrides,
            )
            
            self._call_active = True
            self._logger.info(f"VAPI call started: {self._call_id}")
            
            return self._call_id

        except Exception as e:
            self._logger.error(f"Failed to start VAPI call: {e}", exc_info=True)
            if self._on_error:
                await self._on_error(str(e))
            raise

    def _start_call_blocking(
        self,
        assistant_id: Optional[str] = None,
        assistant: Optional[dict] = None,
        assistant_overrides: Optional[dict] = None,
    ) -> None:
        """
        Blocking call to start VAPI call (runs in executor).
        
        This method is run in a thread executor to avoid blocking the async loop.
        """
        try:
            self._vapi.start(
                assistant_id=assistant_id,
                assistant=assistant,
                assistant_overrides=assistant_overrides,
            )
        except Exception as e:
            self._logger.error(f"Error in blocking call start: {e}", exc_info=True)
            raise

    async def stop_call(self) -> None:
        """Stop the current VAPI call."""
        if not self._call_active:
            self._logger.warning("No active call to stop")
            return

        try:
            self._logger.info("Stopping VAPI call")
            
            # Stop audio streaming
            if self._stream_task and not self._stream_task.done():
                self._stream_task.cancel()
                try:
                    await self._stream_task
                except asyncio.CancelledError:
                    pass
            
            # Stop VAPI call in executor
            if self._vapi:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._vapi.stop)
                self._vapi = None
            
            self._call_active = False
            self._call_id = None
            
            self._logger.info("VAPI call stopped")

        except Exception as e:
            self._logger.error(f"Error stopping VAPI call: {e}", exc_info=True)
            if self._on_error:
                await self._on_error(str(e))

    async def stream_audio(self, audio_data: bytes) -> None:
        """
        Stream audio data to VAPI.
        
        This method queues audio data for streaming to the VAPI call.
        The actual streaming happens in a background task.
        
        Args:
            audio_data: Raw PCM audio data (16-bit, 16kHz, mono)
        """
        if not self._call_active:
            self._logger.warning("Cannot stream audio: no active call")
            return

        try:
            await self._audio_queue.put(audio_data)
        except Exception as e:
            self._logger.error(f"Error queueing audio data: {e}", exc_info=True)

    async def send_message(self, role: str, content: str) -> None:
        """
        Send a text message to the assistant.
        
        Args:
            role: Message role (e.g., "user", "assistant")
            content: Message content
        """
        if not self._call_active or not self._vapi:
            self._logger.warning("Cannot send message: no active call")
            return

        try:
            self._logger.debug(f"Sending message: role={role}, content={content[:50]}...")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._vapi.add_message,
                role,
                content,
            )
            
        except Exception as e:
            self._logger.error(f"Error sending message: {e}", exc_info=True)
            if self._on_error:
                await self._on_error(str(e))

    async def _process_audio_queue(self) -> None:
        """
        Background task to process audio queue and stream to VAPI.
        
        This task runs continuously while a call is active, streaming
        audio data from the queue to VAPI.
        """
        self._logger.debug("Audio streaming task started")
        
        try:
            while self._call_active:
                try:
                    # Wait for audio data with timeout
                    _ = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )
                    
                    # Stream to VAPI (this needs to be adapted based on actual VAPI audio API)
                    # For now, we'll use the Daily.co integration that VAPI uses
                    # The actual implementation will depend on how VAPI exposes audio streaming
                    if self._vapi and hasattr(self._vapi, '_client'):
                        # Access the underlying Daily client if available
                        # This is a placeholder - actual implementation depends on VAPI internals
                        pass
                    
                except asyncio.TimeoutError:
                    # No audio data available, continue
                    continue
                except Exception as e:
                    self._logger.error(f"Error streaming audio: {e}", exc_info=True)
                    
        except asyncio.CancelledError:
            self._logger.debug("Audio streaming task cancelled")
        except Exception as e:
            self._logger.error(f"Audio streaming task error: {e}", exc_info=True)
        finally:
            self._logger.debug("Audio streaming task stopped")
