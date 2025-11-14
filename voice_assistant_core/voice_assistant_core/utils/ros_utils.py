"""
ROS2 utility functions for voice assistant core.

This module provides utilities for converting between internal data types
and ROS2 message types, enabling seamless integration with voice_assistant_msgs.
"""

import json
import time
from typing import Dict, Any, Optional
from builtin_interfaces.msg import Time

try:
    # Import voice_assistant_msgs if available for typed messages
    from voice_assistant_msgs.msg import (
        AudioChunk as AudioChunkMsg,
        AssistantState as AssistantStateMsg,
        VoiceEvent as VoiceEventMsg,
    )

    TYPED_MSGS_AVAILABLE = True
except ImportError:
    # Fallback to std_msgs for compatibility
    TYPED_MSGS_AVAILABLE = False

    # Define placeholder classes to avoid import errors
    class AudioChunkMsg:
        pass

    class AssistantStateMsg:
        pass

    class VoiceEventMsg:
        pass


from std_msgs.msg import String
from ..audio.types import (
    AudioChunk,
    AssistantState,
    VoiceEvent,
    TranscriptionResult,
    LLMResponse,
)


def seconds_to_ros_time(seconds: float) -> Time:
    """
    Convert seconds timestamp to ROS2 Time message.

    Args:
        seconds: Unix timestamp in seconds

    Returns:
        ROS2 Time message
    """
    ros_time = Time()
    ros_time.sec = int(seconds)
    ros_time.nanosec = int((seconds - ros_time.sec) * 1e9)
    return ros_time


def ros_time_to_seconds(ros_time: Time) -> float:
    """
    Convert ROS2 Time message to seconds timestamp.

    Args:
        ros_time: ROS2 Time message

    Returns:
        Unix timestamp in seconds
    """
    return float(ros_time.sec) + (float(ros_time.nanosec) / 1e9)


def audio_chunk_to_msg(chunk: AudioChunk) -> AudioChunkMsg:
    """
    Convert internal AudioChunk to ROS2 message.

    Args:
        chunk: Internal AudioChunk dataclass

    Returns:
        ROS2 AudioChunk message

    Raises:
        ImportError: If voice_assistant_msgs not available
    """
    if not TYPED_MSGS_AVAILABLE:
        raise ImportError("voice_assistant_msgs not available for typed messages")

    msg = AudioChunkMsg()
    msg.data = list(chunk.data)  # Convert bytes to uint8 array
    msg.timestamp = seconds_to_ros_time(chunk.timestamp)
    msg.sequence_id = chunk.sequence_id
    msg.format = chunk.format.name
    msg.sample_rate = chunk.sample_rate
    msg.channels = chunk.channels
    msg.sample_width = chunk.sample_width
    msg.is_final = chunk.is_final

    return msg


def audio_chunk_to_string(chunk: AudioChunk) -> String:
    """
    Convert internal AudioChunk to String message (fallback).

    Args:
        chunk: Internal AudioChunk dataclass

    Returns:
        ROS2 String message with JSON-encoded data
    """
    chunk_dict = {
        "data": chunk.data.hex(),  # Convert bytes to hex string for JSON
        "timestamp": chunk.timestamp,
        "sequence_id": chunk.sequence_id,
        "format": chunk.format.name,
        "sample_rate": chunk.sample_rate,
        "channels": chunk.channels,
        "sample_width": chunk.sample_width,
        "is_final": chunk.is_final,
    }

    msg = String()
    msg.data = json.dumps(chunk_dict)
    return msg


def assistant_state_to_msg(
    state: AssistantState,
    previous_state: Optional[AssistantState] = None,
    state_data: str = "",
) -> AssistantStateMsg:
    """
    Convert internal AssistantState to ROS2 message.

    Args:
        state: Current assistant state
        previous_state: Previous state (optional)
        state_data: Additional state information

    Returns:
        ROS2 AssistantState message

    Raises:
        ImportError: If voice_assistant_msgs not available
    """
    if not TYPED_MSGS_AVAILABLE:
        raise ImportError("voice_assistant_msgs not available for typed messages")

    msg = AssistantStateMsg()
    msg.current_state = state.name
    msg.previous_state = previous_state.name if previous_state else ""
    msg.transition_time = seconds_to_ros_time(time.time())
    msg.state_data = state_data

    return msg


def assistant_state_to_string(
    state: AssistantState,
    previous_state: Optional[AssistantState] = None,
    state_data: str = "",
) -> String:
    """
    Convert internal AssistantState to String message (fallback).

    Args:
        state: Current assistant state
        previous_state: Previous state (optional)
        state_data: Additional state information

    Returns:
        ROS2 String message with JSON-encoded data
    """
    import time

    state_dict = {
        "current_state": state.name,
        "previous_state": previous_state.name if previous_state else "",
        "transition_time": time.time(),
        "state_data": state_data,
    }

    msg = String()
    msg.data = json.dumps(state_dict)
    return msg


def voice_event_to_msg(event: VoiceEvent) -> VoiceEventMsg:
    """
    Convert internal VoiceEvent to ROS2 message.

    Args:
        event: Internal VoiceEvent dataclass

    Returns:
        ROS2 VoiceEvent message

    Raises:
        ImportError: If voice_assistant_msgs not available
    """
    if not TYPED_MSGS_AVAILABLE:
        raise ImportError("voice_assistant_msgs not available for typed messages")

    msg = VoiceEventMsg()
    msg.event_type = event.event_type.name
    msg.message = event.message
    msg.timestamp = seconds_to_ros_time(event.timestamp)
    msg.priority = event.priority
    msg.event_data = json.dumps(event.event_data) if event.event_data else ""

    return msg


def voice_event_to_string(event: VoiceEvent) -> String:
    """
    Convert internal VoiceEvent to String message (fallback).

    Args:
        event: Internal VoiceEvent dataclass

    Returns:
        ROS2 String message with JSON-encoded data
    """
    event_dict = {
        "event_type": event.event_type.name,
        "message": event.message,
        "timestamp": event.timestamp,
        "priority": event.priority,
        "event_data": event.event_data or {},
    }

    msg = String()
    msg.data = json.dumps(event_dict)
    return msg


def string_to_transcription_result(msg: String) -> TranscriptionResult:
    """
    Convert String message to internal TranscriptionResult (fallback parsing).

    Args:
        msg: ROS2 String message with JSON-encoded data

    Returns:
        Internal TranscriptionResult dataclass

    Raises:
        ValueError: If message format is invalid
    """
    try:
        data = json.loads(msg.data)
        return TranscriptionResult(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            language=data.get("language", "en"),
            processing_time=data.get("processing_time", 0.0),
            error_message=data.get("error_message"),
            audio_duration=data.get("audio_duration", 0.0),
        )
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid transcription result message format: {e}")


def string_to_llm_response(msg: String) -> LLMResponse:
    """
    Convert String message to internal LLMResponse (fallback parsing).

    Args:
        msg: ROS2 String message with JSON-encoded data

    Returns:
        Internal LLMResponse dataclass

    Raises:
        ValueError: If message format is invalid
    """
    try:
        data = json.loads(msg.data)
        return LLMResponse(
            response_text=data.get("response_text", ""),
            intent=data.get("intent", ""),
            confidence=data.get("confidence", 0.0),
            continue_conversation=data.get("continue_conversation", False),
            conversation_id=data.get("conversation_id", ""),
            entities=data.get("entities", []),
            processing_time=data.get("processing_time", 0.0),
            error_message=data.get("error_message"),
        )
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid LLM response message format: {e}")


def get_message_type_info() -> Dict[str, Any]:
    """
    Get information about available message types.

    Returns:
        Dictionary with message type availability and versions
    """
    return {
        "typed_messages_available": TYPED_MSGS_AVAILABLE,
        "voice_assistant_msgs_available": TYPED_MSGS_AVAILABLE,
        "fallback_mode": not TYPED_MSGS_AVAILABLE,
        "supported_conversions": [
            "AudioChunk -> AudioChunkMsg/String",
            "AssistantState -> AssistantStateMsg/String",
            "VoiceEvent -> VoiceEventMsg/String",
            "String -> TranscriptionResult",
            "String -> LLMResponse",
        ],
    }
