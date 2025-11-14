"""
Audio buffer management for voice assistant core.

Simplified circular buffer with continuous streaming support.
All turn detection logic has been moved to TurnDetector class.

The AudioBuffer class provides:
- Thread-safe circular audio chunk buffering
- Configurable maximum duration with dynamic deque sizing
- Simple stream management without boundary markers
- Efficient memory management for Raspberry Pi 5
- Statistics and monitoring

ROS2 Integration:
- Designed to work with voice_assistant_msgs/AudioChunk messages
- Compatible with ROS2 QoS policies and timing requirements
"""

import asyncio
import logging
import time
import wave
import io
from typing import Optional, Callable, AsyncIterator
from collections import deque
from dataclasses import dataclass

from .types import AudioChunk, AudioFormat
from ..constants import (
    MAX_AUDIO_BUFFER_SIZE,
    AUDIO_CHUNK_SIZE,
    MAX_AUDIO_DURATION_SEC,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLE_WIDTH,
    DEFAULT_CHANNELS,
    AUDIO_QUEUE_MAX_SIZE,
)


@dataclass
class AudioStreamStats:
    """Statistics for audio stream processing."""

    total_chunks: int = 0
    total_bytes: int = 0
    stream_duration: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    sample_width: int = 0
    is_active: bool = False


class AudioBufferError(Exception):
    """Raised when audio buffer operations fail."""

    pass


class AudioFormatError(Exception):
    """Raised when audio format validation fails."""

    pass


class AudioBuffer:
    """
    Simplified audio buffer with circular deque for continuous streaming.

    All turn boundary logic has been moved to TurnDetector class.
    This class now only handles raw audio buffering and streaming.
    """

    def __init__(
        self,
        max_buffer_size: int = MAX_AUDIO_BUFFER_SIZE,
        max_duration: float = MAX_AUDIO_DURATION_SEC,
        log_level: str = "WARNING",
    ) -> None:
        """
        Initialize audio buffer.

        Args:
            max_buffer_size: Maximum buffer size in bytes (default: 1MB)
            max_duration: Maximum audio buffer duration in seconds (default: 30s)
            log_level: Logging level (WARNING for production, INFO for development)
        """
        if max_buffer_size <= 0:
            raise ValueError(f"max_buffer_size must be positive, got {max_buffer_size}")
        if max_duration <= 0:
            raise ValueError(f"max_duration must be positive, got {max_duration}")

        self._max_buffer_size = max_buffer_size
        self._max_duration = max_duration

        # Circular buffer with automatic size calculation
        # Will be resized in start_stream() based on audio format
        self._chunks: deque = deque(maxlen=AUDIO_QUEUE_MAX_SIZE)
        self._buffer_size = 0

        # Stream state
        self._is_streaming = False
        self._stream_id = 0
        self._sequence_counter = 0

        # Audio format tracking
        self._current_format: Optional[AudioFormat] = None
        self._sample_rate = DEFAULT_SAMPLE_RATE
        self._sample_width = DEFAULT_SAMPLE_WIDTH
        self._channels = DEFAULT_CHANNELS

        # Statistics
        self._stats = AudioStreamStats()

        # Callbacks
        self._stream_start_callback: Optional[Callable] = None
        self._stream_end_callback: Optional[Callable] = None
        self._chunk_callback: Optional[Callable] = None

        # Async coordination
        self._stream_complete_event = asyncio.Event()
        self._new_chunk_event = asyncio.Event()

        # Logger
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))

        self._logger.info("Audio buffer initialized (simplified circular buffer)")

    @property
    def is_streaming(self) -> bool:
        """Check if audio stream is active."""
        return self._is_streaming

    @property
    def buffer_size(self) -> int:
        """Get current buffer size in bytes."""
        return self._buffer_size

    @property
    def chunk_count(self) -> int:
        """Get number of chunks in buffer."""
        return len(self._chunks)

    @property
    def stream_stats(self) -> AudioStreamStats:
        """Get current stream statistics."""
        return self._stats

    def start_stream(
        self,
        audio_format: AudioFormat = AudioFormat.PCM_16KHZ_16BIT_MONO,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
        channels: int = DEFAULT_CHANNELS,
    ) -> int:
        """
        Start a new audio stream.

        Args:
            audio_format: Audio format type
            sample_rate: Audio sample rate in Hz
            sample_width: Bits per sample
            channels: Number of audio channels

        Returns:
            Stream ID for tracking
        """
        if self._is_streaming:
            raise AudioBufferError("Audio stream already active")

        # Clear previous data
        self._chunks.clear()
        self._buffer_size = 0
        self._sequence_counter = 0

        # Set stream parameters
        self._current_format = audio_format
        self._sample_rate = sample_rate
        self._sample_width = sample_width
        self._channels = channels

        # Calculate optimal deque maxlen based on max_duration and chunk size
        # Assume typical chunk size of AUDIO_CHUNK_SIZE bytes
        bytes_per_second = sample_rate * channels * (sample_width // 8)
        max_buffer_bytes = int(self._max_duration * bytes_per_second)
        optimal_maxlen = max(
            int(max_buffer_bytes / AUDIO_CHUNK_SIZE), AUDIO_QUEUE_MAX_SIZE
        )

        # Recreate deque with optimal size for memory efficiency
        self._chunks = deque(maxlen=optimal_maxlen)

        self._logger.debug(
            f"Circular buffer sized for {self._max_duration}s "
            f"(maxlen={optimal_maxlen} chunks)"
        )

        # Initialize statistics
        self._stream_id += 1
        self._stats = AudioStreamStats(
            start_time=time.time(),
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            is_active=True,
        )

        self._is_streaming = True
        self._stream_complete_event.clear()

        self._logger.info(
            f"Audio stream started - ID: {self._stream_id}, "
            f"Format: {audio_format.name}, Rate: {sample_rate}Hz"
        )

        # Call stream start callback
        if self._stream_start_callback:
            try:
                self._stream_start_callback(self._stream_id, self._stats)
            except Exception as e:
                self._logger.error(f"Error in stream start callback: {e}")

        return self._stream_id

    def add_chunk(
        self, data: bytes, timestamp: Optional[float] = None, is_final: bool = False
    ) -> bool:
        """
        Add audio chunk to circular buffer.

        Args:
            data: Raw audio data bytes
            timestamp: Chunk timestamp (uses time.time() if None)
            is_final: True if this is the last chunk (ends stream)

        Returns:
            True if chunk added successfully
        """
        if data is None or len(data) == 0:
            raise ValueError("Audio data cannot be None or empty")
        if not self._is_streaming:
            raise AudioBufferError("No active audio stream")

        if timestamp is None:
            timestamp = time.time()

        # Validate audio chunk
        self._validate_audio_chunk(data)

        # Create audio chunk
        chunk = AudioChunk(
            data=data,
            timestamp=timestamp,
            sequence_id=self._sequence_counter,
            format=self._current_format,
            sample_rate=self._sample_rate,
            channels=self._channels,
            sample_width=self._sample_width,
            is_final=is_final,
        )

        # Add to circular buffer (automatically drops oldest if full)
        self._chunks.append(chunk)
        self._buffer_size += len(data)
        self._sequence_counter += 1

        # Maintain accurate buffer size by subtracting dropped chunks
        # (deque automatically drops oldest when maxlen reached)
        if len(self._chunks) == self._chunks.maxlen:
            # Buffer is at max capacity, oldest was dropped
            self._buffer_size = sum(len(c.data) for c in self._chunks)

        # Update statistics
        self._stats.total_chunks += 1
        self._stats.total_bytes += len(data)
        self._stats.stream_duration = timestamp - self._stats.start_time

        # Call chunk callback
        if self._chunk_callback:
            try:
                self._chunk_callback(chunk, self._stats)
            except Exception as e:
                self._logger.error(f"Error in chunk callback: {e}")

        # Signal new chunk available
        self._new_chunk_event.set()
        self._new_chunk_event.clear()

        # End stream if final chunk
        if is_final:
            self.end_stream()

        return True

    def get_recent_audio(self, max_duration_secs: Optional[float] = None) -> bytes:
        """
        Get recent audio from circular buffer.

        Args:
            max_duration_secs: Maximum duration in seconds (None = all buffer)

        Returns:
            Audio data as bytes
        """
        if not self._chunks:
            return b""

        chunks_list = list(self._chunks)

        if max_duration_secs is None:
            # Return all buffer
            audio_data = b"".join(chunk.data for chunk in chunks_list)
        else:
            # Calculate how many chunks fit in duration
            bytes_per_second = (
                self._sample_rate * self._channels * (self._sample_width // 8)
            )
            max_bytes = int(max_duration_secs * bytes_per_second)

            # Take chunks from end until we reach max_bytes
            audio_data = b""
            for chunk in reversed(chunks_list):
                if len(audio_data) + len(chunk.data) > max_bytes:
                    break
                audio_data = chunk.data + audio_data

        duration = len(audio_data) / (
            self._sample_rate * self._channels * (self._sample_width // 8)
        )
        self._logger.debug(
            f"Extracted {len(audio_data)} bytes ({duration:.2f}s) from buffer"
        )

        return audio_data

    def end_stream(self) -> bool:
        """
        End the current audio stream.

        Returns:
            True if stream ended successfully
        """
        if not self._is_streaming:
            self._logger.warning("No active stream to end")
            return False

        self._is_streaming = False
        self._stats.end_time = time.time()
        self._stats.is_active = False

        # Update final duration
        if self._stats.start_time > 0:
            self._stats.stream_duration = self._stats.end_time - self._stats.start_time

        self._logger.info(
            f"Audio stream ended - ID: {self._stream_id}, "
            f"Duration: {self._stats.stream_duration:.2f}s, "
            f"Chunks: {self._stats.total_chunks}, "
            f"Bytes: {self._stats.total_bytes}"
        )

        # Signal stream complete
        self._stream_complete_event.set()

        # Call stream end callback
        if self._stream_end_callback:
            try:
                self._stream_end_callback(self._stream_id, self._stats)
            except Exception as e:
                self._logger.error(f"Error in stream end callback: {e}")

        return True

    async def get_audio_stream(self) -> AsyncIterator[AudioChunk]:
        """
        Get async iterator for audio chunks.

        Yields:
            AudioChunk objects as they become available
        """
        while self._is_streaming or len(self._chunks) > 0:
            if len(self._chunks) == 0:
                if not self._is_streaming:
                    break
                # Wait for new chunk or stream end
                try:
                    await asyncio.wait_for(self._new_chunk_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

            if len(self._chunks) > 0:
                chunk = self._chunks.popleft()
                self._buffer_size -= len(chunk.data)
                yield chunk

                if chunk.is_final:
                    break

    async def get_complete_audio(
        self, timeout: float = MAX_AUDIO_DURATION_SEC
    ) -> bytes:
        """
        Get complete audio stream as bytes.

        Args:
            timeout: Maximum time to wait for stream completion

        Returns:
            Complete audio data as bytes
        """
        if not self._is_streaming and len(self._chunks) == 0:
            raise AudioBufferError("No active audio stream")

        # Wait for stream to complete
        try:
            await asyncio.wait_for(self._stream_complete_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._logger.error(f"Audio stream timeout after {timeout}s")
            raise

        # Combine all chunks
        audio_data = b"".join(chunk.data for chunk in self._chunks)

        self._logger.debug(f"Complete audio assembled - {len(audio_data)} bytes")
        return audio_data

    def get_wav_audio(self) -> bytes:
        """
        Get complete audio as WAV format bytes.

        Returns:
            WAV formatted audio data
        """
        if len(self._chunks) == 0:
            raise AudioBufferError("No audio data available")

        # Combine all chunk data
        audio_data = b"".join(chunk.data for chunk in self._chunks)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self._channels)
            wav_file.setsampwidth(self._sample_width // 8)
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(audio_data)

        wav_data = wav_buffer.getvalue()
        wav_buffer.close()

        self._logger.debug(f"WAV audio created - {len(wav_data)} bytes")
        return wav_data

    def clear_buffer(self) -> None:
        """Clear all buffered audio data and reset state."""
        self._chunks.clear()
        self._buffer_size = 0
        self._sequence_counter = 0

        if self._is_streaming:
            self.end_stream()

        self._logger.debug("Audio buffer cleared")

    def _validate_audio_chunk(self, data: bytes) -> None:
        """
        Validate audio chunk data.

        Args:
            data: Raw audio data bytes to validate
        """
        if not data:
            raise AudioFormatError("Empty audio data not allowed")

        # Warn on oversized chunks
        if len(data) > AUDIO_CHUNK_SIZE * 10:
            self._logger.warning(
                f"Large audio chunk: {len(data)} bytes (>{AUDIO_CHUNK_SIZE * 10})"
            )

        # Validate frame alignment
        if self._current_format == AudioFormat.PCM_16KHZ_16BIT_MONO:
            frame_size = (self._sample_width // 8) * self._channels
            if len(data) % frame_size != 0:
                self._logger.warning(
                    f"Audio chunk ({len(data)} bytes) not frame-aligned "
                    f"(frame_size={frame_size})"
                )

    def set_stream_start_callback(self, callback: Callable) -> None:
        """Set callback for stream start events."""
        self._stream_start_callback = callback

    def set_stream_end_callback(self, callback: Callable) -> None:
        """Set callback for stream end events."""
        self._stream_end_callback = callback

    def set_chunk_callback(self, callback: Callable) -> None:
        """Set callback for new chunk events."""
        self._chunk_callback = callback

    def get_buffer_info(self) -> dict:
        """
        Get comprehensive buffer information for diagnostics.

        Returns:
            Dictionary with buffer status and metrics
        """
        current_time = time.time()
        buffer_utilization = (self._buffer_size / self._max_buffer_size) * 100.0

        throughput_bps = 0.0
        if self._stats.stream_duration > 0:
            throughput_bps = self._stats.total_bytes / self._stats.stream_duration

        return {
            "is_streaming": self._is_streaming,
            "stream_id": self._stream_id,
            "buffer_size_bytes": self._buffer_size,
            "chunk_count": len(self._chunks),
            "sequence_counter": self._sequence_counter,
            "current_format": self._current_format.name
            if self._current_format
            else None,
            "sample_rate": self._sample_rate,
            "sample_width": self._sample_width,
            "channels": self._channels,
            "max_buffer_size": self._max_buffer_size,
            "max_duration": self._max_duration,
            "buffer_utilization_percent": round(buffer_utilization, 2),
            "throughput_bytes_per_second": round(throughput_bps, 2),
            "stats": {
                "total_chunks": self._stats.total_chunks,
                "total_bytes": self._stats.total_bytes,
                "stream_duration": round(self._stats.stream_duration, 3),
                "is_active": self._stats.is_active,
                "start_time": self._stats.start_time,
                "end_time": self._stats.end_time
                if self._stats.end_time > 0
                else current_time,
            },
        }
