"""
Turn detection module for voice assistant core.

This module implements a two-phase end-of-turn detection system:
1. WAITING_FOR_SPEECH_START: Accumulate consecutive speech frames until threshold
2. WAITING_FOR_TURN_END: Analyze during prolonged silence + fallback timeout

Key features:
- Pre-buffering (0.2s) to capture audio before speech detection
- Circular deque buffer with dynamic size (8s context)
- Async ML inference with cancellation support
- Fallback timeout (3s) for guaranteed turn completion
- Optimized for Raspberry Pi 5 with minimal memory footprint
"""

import asyncio
import logging
import os
import time
from collections import deque
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor


class TurnDetectorState(Enum):
    """Turn detector states for two-phase detection."""

    WAITING_FOR_SPEECH_START = "waiting_for_speech_start"
    WAITING_FOR_TURN_END = "waiting_for_turn_end"


class TurnDetector:
    """
    Two-phase turn detector with intelligent speech boundary detection.

    Phase 1: Wait for consecutive speech burst (0.6-0.8s)
    Phase 2: Analyze during silence (1.2-1.5s) with 3s fallback timeout

    Features:
    - Pre-buffer (0.2s) to capture audio before speech starts
    - Continuous circular buffer (8s max) with recent audio
    - Async cancellation if speech detected during ML analysis
    - Model warmup on initialization for low latency
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        speech_start_consecutive_duration: float = 0.7,
        silence_duration_before_analysis: float = 1.5,
        silence_fallback_timeout: float = 3.0,
        context_buffer_duration: float = 8.0,
        pre_buffer_duration: float = 0.2,
        ml_confidence_threshold: float = 0.82,
        min_turn_cooldown: float = 2.5,
        sample_rate: int = 16000,
        chunk_size: int = 2048,
        enabled: bool = True,
        enable_model_warmup: bool = True,
        log_level: str = "INFO",
    ) -> None:
        """
        Initialize two-phase turn detector.

        Args:
            model_path: Path to smart-turn ONNX model
            speech_start_consecutive_duration: Consecutive speech duration to trigger turn start (s)
            silence_duration_before_analysis: Silence duration before ML analysis (s)
            silence_fallback_timeout: Max silence before forced turn end (s)
            context_buffer_duration: Max audio context buffer duration (s)
            pre_buffer_duration: Pre-buffer duration before speech (s)
            ml_confidence_threshold: ML model confidence threshold (0-1)
            min_turn_cooldown: Minimum time between turn completions (s)
            sample_rate: Audio sample rate (must be 16kHz)
            chunk_size: Audio chunk size in bytes
            enabled: Enable turn detection
            enable_model_warmup: Run warmup inference on init
            log_level: Logging level (INFO/WARNING)
        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        self._enabled = enabled
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size

        # Configuration parameters
        self._speech_start_duration = speech_start_consecutive_duration
        self._silence_analysis_duration = silence_duration_before_analysis
        self._silence_fallback_timeout = silence_fallback_timeout
        self._context_buffer_duration = context_buffer_duration
        self._pre_buffer_duration = pre_buffer_duration
        self._ml_threshold = ml_confidence_threshold
        self._min_turn_cooldown = min_turn_cooldown
        self._enable_warmup = enable_model_warmup

        # Calculate buffer sizes
        samples_per_chunk = chunk_size // 2  # 16-bit = 2 bytes per sample
        self._context_buffer_maxlen = int(
            (context_buffer_duration * sample_rate) / samples_per_chunk
        )
        self._prebuffer_maxlen = int(
            (pre_buffer_duration * sample_rate) / samples_per_chunk
        )

        self._logger.info(
            f"Turn detector initialized - speech_start: {speech_start_consecutive_duration}s, "
            f"silence_analysis: {silence_duration_before_analysis}s, "
            f"silence_fallback: {silence_fallback_timeout}s, "
            f"context_buffer: {context_buffer_duration}s ({self._context_buffer_maxlen} chunks), "
            f"pre_buffer: {pre_buffer_duration}s ({self._prebuffer_maxlen} chunks), "
            f"ml_threshold: {ml_confidence_threshold}, "
            f"cooldown: {min_turn_cooldown}s"
        )

        # State management
        self._state = TurnDetectorState.WAITING_FOR_SPEECH_START
        self._consecutive_speech_duration = 0.0
        self._silence_duration = 0.0
        self._last_speech_timestamp = 0.0
        self._last_turn_completion_time = 0.0
        self._turn_start_time = 0.0

        # Audio buffers
        # Pre-buffer: circular buffer for audio before speech detection
        self._prebuffer: deque = deque(maxlen=self._prebuffer_maxlen)
        # Context buffer: circular buffer for speech audio during turn
        self._context_buffer: deque = deque(maxlen=self._context_buffer_maxlen)

        # Async analysis task management
        self._analysis_task: Optional[asyncio.Task] = None
        self._analysis_cancelled_count = 0
        self._analysis_in_progress = (
            False  # Flag to prevent multiple concurrent analyses
        )

        # Model components
        self._session = None
        self._feature_extractor = None
        self._model_warmed_up = False

        # Statistics
        self._total_predictions = 0
        self._turn_completions_detected = 0
        self._fallback_triggered_count = 0

        if self._enabled:
            self._initialize_model(model_path)

    @property
    def enabled(self) -> bool:
        """Check if turn detector is enabled."""
        return self._enabled

    @property
    def state(self) -> TurnDetectorState:
        """Get current detector state."""
        return self._state

    @property
    def turn_in_progress(self) -> bool:
        """Check if turn is in progress (phase 2)."""
        return self._state == TurnDetectorState.WAITING_FOR_TURN_END

    def _initialize_model(self, model_path: Optional[str] = None) -> None:
        """Initialize ONNX model and feature extractor."""
        try:
            if model_path is None:
                # Try default location
                model_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "smart-turn-v3.0.onnx",
                )

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at: {model_path}")

            # Initialize ONNX session with optimizations for Raspberry Pi
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = 2  # Pi 5 can handle 2 threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                model_path, sess_options=so, providers=["CPUExecutionProvider"]
            )

            # Initialize Whisper feature extractor (lazy loaded)
            self._feature_extractor = WhisperFeatureExtractor(chunk_length=8)

            self._logger.info(f"Smart-turn model loaded from: {model_path}")

        except Exception as e:
            self._logger.error(f"Failed to initialize model: {e}")
            self._enabled = False
            raise

    async def warm_up(self) -> None:
        """
        Warm up model with dummy inference to optimize first real prediction.

        First inference on Raspberry Pi can take 200-500ms extra due to
        model loading and optimization. This runs a dummy prediction to
        pre-warm the model.
        """
        if not self._enabled or self._model_warmed_up:
            return

        try:
            self._logger.info("Warming up turn detector model...")

            # Create 1 second of silent audio
            dummy_audio = np.zeros(self._sample_rate, dtype=np.float32)

            # Run inference in thread to not block
            await asyncio.to_thread(self._predict_endpoint, dummy_audio)

            self._model_warmed_up = True
            self._logger.info("Model warmup completed")

        except Exception as e:
            self._logger.warning(f"Model warmup failed: {e}")

    def reset_stream(self) -> None:
        """Reset detector state for new audio stream."""
        # Cancel pending analysis
        if self._analysis_task and not self._analysis_task.done():
            self._analysis_task.cancel()

        self._state = TurnDetectorState.WAITING_FOR_SPEECH_START
        self._consecutive_speech_duration = 0.0
        self._silence_duration = 0.0
        self._last_speech_timestamp = 0.0
        self._turn_start_time = 0.0

        self._prebuffer.clear()
        self._context_buffer.clear()

        self._analysis_task = None
        self._analysis_in_progress = False  # Reset analysis flag

        self._logger.debug("Turn detector stream reset")

    def add_audio_chunk(
        self, audio_data: bytes, is_speech: bool, timestamp: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Add audio chunk and update detection state.

        Two-phase processing:
        1. WAITING_FOR_SPEECH_START: Accumulate consecutive speech
        2. WAITING_FOR_TURN_END: Track silence and trigger analysis

        Args:
            audio_data: Raw audio (16-bit PCM)
            is_speech: VAD speech detection result
            timestamp: Optional timestamp (uses time.time() if None)

        Returns:
            Tuple (state_changed: bool, event: Optional[str])
            - state_changed: True if state transitioned
            - event: "turn_started" | "silence_detected" | "fallback_timeout" | None
        """
        if not self._enabled:
            return False, None

        try:
            if timestamp is None:
                timestamp = time.time()

            # Convert to float32 numpy array
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            chunk_duration = len(audio_int16) / self._sample_rate

            # Phase 1: Waiting for speech start
            if self._state == TurnDetectorState.WAITING_FOR_SPEECH_START:
                # Always maintain pre-buffer (circular)
                self._prebuffer.append(audio_float32)

                if is_speech:
                    self._consecutive_speech_duration += chunk_duration

                    # Transition to phase 2 when consecutive speech threshold reached
                    if self._consecutive_speech_duration >= self._speech_start_duration:
                        self._state = TurnDetectorState.WAITING_FOR_TURN_END
                        self._turn_start_time = timestamp
                        self._last_speech_timestamp = timestamp

                        # Transfer pre-buffer to context buffer
                        for chunk in self._prebuffer:
                            self._context_buffer.append(chunk)

                        self._logger.debug(
                            f"Turn started - consecutive speech: {self._consecutive_speech_duration:.2f}s"
                        )

                        return True, "turn_started"
                else:
                    # Reset consecutive speech counter on silence
                    self._consecutive_speech_duration = 0.0

                return False, None

            # Phase 2: Waiting for turn end
            elif self._state == TurnDetectorState.WAITING_FOR_TURN_END:
                # Always add to context buffer (circular, maintains last N seconds)
                self._context_buffer.append(audio_float32)

                if is_speech:
                    # Reset silence tracking on speech
                    self._silence_duration = 0.0
                    self._last_speech_timestamp = timestamp

                    # Cancel pending analysis if speech detected during processing
                    if self._analysis_task and not self._analysis_task.done():
                        self._analysis_task.cancel()
                        self._analysis_cancelled_count += 1
                        self._analysis_in_progress = False  # Reset flag
                        self._logger.debug(
                            f"Analysis cancelled due to new speech (count: {self._analysis_cancelled_count})"
                        )

                    return False, None
                else:
                    # Accumulate silence duration
                    self._silence_duration += chunk_duration

                    # Check fallback timeout (3s since last speech)
                    time_since_last_speech = timestamp - self._last_speech_timestamp
                    if time_since_last_speech >= self._silence_fallback_timeout:
                        self._fallback_triggered_count += 1
                        self._logger.debug(
                            f"Fallback timeout triggered - {time_since_last_speech:.2f}s since last speech"
                        )
                        return False, "fallback_timeout"

                    # Trigger analysis after silence threshold (if not already analyzing)
                    if (
                        self._silence_duration >= self._silence_analysis_duration
                        and not self._analysis_in_progress
                    ):
                        self._analysis_in_progress = True
                        self._logger.debug(
                            f"Silence detected - triggering analysis ({self._silence_duration:.2f}s)"
                        )
                        return False, "silence_detected"

                    return False, None

            return False, None

        except Exception as e:
            self._logger.error(f"Error in add_audio_chunk: {e}")
            return False, None

    async def analyze_end_of_turn(
        self, force: bool = False
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Analyze accumulated audio to determine if turn is complete.

        This creates an async task for ML inference. The task can be cancelled
        if new speech is detected via add_audio_chunk().

        Args:
            force: Force turn completion (used for fallback timeout)

        Returns:
            Tuple (is_complete: bool, result: Optional[Dict])
            - is_complete: True if turn is complete
            - result: {'prediction': 0|1, 'probability': float, 'inference_time_ms': float}
        """
        if not self._enabled:
            return False, None

        # Force completion for fallback timeout
        if force:
            self._complete_turn()
            return True, {
                "prediction": 1,
                "probability": 1.0,
                "inference_time_ms": 0.0,
                "forced": True,
            }

        # Check cooldown to prevent rapid consecutive completions
        current_time = time.time()
        time_since_last = current_time - self._last_turn_completion_time
        if time_since_last < self._min_turn_cooldown:
            self._logger.debug(
                f"Cooldown active - {time_since_last:.2f}s < {self._min_turn_cooldown}s"
            )
            return False, None

        if not self._context_buffer:
            self._logger.warning("Empty context buffer in analyze_end_of_turn")
            return False, None

        try:
            # Extract audio from context buffer
            audio_chunks = list(self._context_buffer)
            segment_audio = np.concatenate(audio_chunks)

            duration_sec = len(segment_audio) / self._sample_rate
            self._logger.debug(
                f"Analyzing turn - {duration_sec:.2f}s ({len(audio_chunks)} chunks)"
            )

            # Run ML inference in thread (non-blocking)
            start_time = time.time()
            result = await asyncio.to_thread(self._predict_endpoint, segment_audio)
            inference_time_ms = (time.time() - start_time) * 1000

            result["inference_time_ms"] = inference_time_ms
            result["forced"] = False

            is_complete = result["prediction"] == 1

            self._total_predictions += 1

            self._logger.debug(
                f"Prediction: {'COMPLETE' if is_complete else 'INCOMPLETE'} "
                f"(prob: {result['probability']:.4f}, time: {inference_time_ms:.1f}ms)"
            )

            if is_complete:
                self._complete_turn()
            else:
                # Analysis complete but turn not finished - allow new analysis
                self._analysis_in_progress = False

            return is_complete, result

        except asyncio.CancelledError:
            self._logger.debug("Analysis cancelled")
            self._analysis_in_progress = False  # Reset on cancellation
            raise
        except Exception as e:
            self._logger.error(f"Error in analyze_end_of_turn: {e}")
            self._analysis_in_progress = False  # Reset on error
            return False, None

    def _complete_turn(self) -> None:
        """
        Complete current turn and reset to phase 1.

        Called when:
        - ML model predicts COMPLETE
        - Fallback timeout triggers
        """
        self._turn_completions_detected += 1
        self._last_turn_completion_time = time.time()

        turn_duration = self._last_turn_completion_time - self._turn_start_time

        self._logger.info(
            f"Turn completed - duration: {turn_duration:.2f}s, "
            f"total completions: {self._turn_completions_detected}"
        )

        # Reset to phase 1
        self._state = TurnDetectorState.WAITING_FOR_SPEECH_START
        self._consecutive_speech_duration = 0.0
        self._silence_duration = 0.0
        self._last_speech_timestamp = 0.0
        self._turn_start_time = 0.0

        # Clear context buffer but maintain pre-buffer
        self._context_buffer.clear()

        # Cancel any pending analysis and reset flag
        if self._analysis_task and not self._analysis_task.done():
            self._analysis_task.cancel()
        self._analysis_task = None
        self._analysis_in_progress = False  # Reset analysis flag

    def get_audio_for_transcription(
        self, include_prebuffer: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract audio from context buffer for transcription.

        Args:
            include_prebuffer: Include pre-buffered audio (0.2s before speech)

        Returns:
            Audio as float32 numpy array, or None if buffer empty
        """
        try:
            if not self._context_buffer:
                return None

            audio_chunks = list(self._context_buffer)
            audio = np.concatenate(audio_chunks)

            duration = len(audio) / self._sample_rate
            self._logger.debug(
                f"Extracted audio for transcription: {duration:.2f}s, "
                f"prebuffer_included: {include_prebuffer}"
            )

            return audio

        except Exception as e:
            self._logger.error(f"Error extracting audio: {e}")
            return None

    def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """
        Run ML inference to predict turn completion.

        Uses Whisper feature extraction + ONNX model inference.
        Optimized for Raspberry Pi 5.

        Args:
            audio_array: Audio as float32 numpy array

        Returns:
            Dict with 'prediction' (0|1) and 'probability' (0-1)
        """
        try:
            # Truncate to last 8 seconds or pad to 8 seconds
            max_samples = 8 * self._sample_rate
            if len(audio_array) > max_samples:
                audio_array = audio_array[-max_samples:]  # Keep last 8s
            elif len(audio_array) < max_samples:
                # Pad with zeros at beginning
                padding = max_samples - len(audio_array)
                audio_array = np.pad(
                    audio_array, (padding, 0), mode="constant", constant_values=0
                )

            # Extract features using Whisper
            inputs = self._feature_extractor(
                audio_array,
                sampling_rate=self._sample_rate,
                return_tensors="np",
                padding="max_length",
                max_length=max_samples,
                truncation=True,
                do_normalize=True,
            )

            # Prepare input for ONNX
            input_features = inputs.input_features.squeeze(0).astype(np.float32)
            input_features = np.expand_dims(input_features, axis=0)

            # Run ONNX inference
            outputs = self._session.run(None, {"input_features": input_features})

            # Extract probability
            probability = outputs[0][0].item()

            # Threshold prediction
            prediction = 1 if probability >= self._ml_threshold else 0

            return {"prediction": prediction, "probability": probability}

        except Exception as e:
            self._logger.error(f"Prediction error: {e}")
            return {"prediction": 0, "probability": 0.0}

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics and telemetry."""
        return {
            "enabled": self._enabled,
            "state": self._state.value,
            "turn_in_progress": self.turn_in_progress,
            "model_warmed_up": self._model_warmed_up,
            "total_predictions": self._total_predictions,
            "turn_completions_detected": self._turn_completions_detected,
            "fallback_triggered_count": self._fallback_triggered_count,
            "analysis_cancelled_count": self._analysis_cancelled_count,
            "context_buffer_chunks": len(self._context_buffer),
            "prebuffer_chunks": len(self._prebuffer),
            "consecutive_speech_duration": self._consecutive_speech_duration,
            "silence_duration": self._silence_duration,
        }

    def cleanup(self) -> None:
        """Release resources and cleanup."""
        try:
            # Cancel pending tasks
            if self._analysis_task and not self._analysis_task.done():
                self._analysis_task.cancel()

            # Clear buffers
            self._context_buffer.clear()
            self._prebuffer.clear()

            # Release model resources
            self._session = None
            self._feature_extractor = None

            self._logger.info("Turn detector cleanup completed")

        except Exception as e:
            self._logger.warning(f"Cleanup error: {e}")
        finally:
            self._enabled = False
