"""
Optimized JSON utilities for voice assistant core.

This module provides optimized JSON encoding/decoding functions
to reduce serialization overhead in high-frequency operations.
"""

import json
from typing import Any, Dict


class OptimizedJSONEncoder(json.JSONEncoder):
    """Optimized JSON encoder with reduced formatting overhead."""

    def __init__(self, *args, **kwargs):
        # Remove separators and indentation for minimal size
        super().__init__(separators=(",", ":"), *args, **kwargs)


# Pre-create encoder instance to avoid repeated instantiation
_encoder = OptimizedJSONEncoder()


def fast_json_dumps(obj: Any) -> str:
    """
    Fast JSON serialization without pretty formatting.

    Args:
        obj: Object to serialize

    Returns:
        Compact JSON string
    """
    return _encoder.encode(obj)


def fast_json_loads(json_str: str) -> Any:
    """
    Fast JSON deserialization.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed object
    """
    return json.loads(json_str)


def create_state_message(
    state_name: str, timestamp: float, time_in_state: float, error_count: int
) -> str:
    """
    Create optimized state message with minimal serialization overhead.

    Args:
        state_name: Current state name
        timestamp: Current timestamp
        time_in_state: Time spent in current state
        error_count: Number of errors encountered

    Returns:
        Compact JSON string
    """
    # Use direct string formatting for better performance on repeated calls
    return f'{{"state":"{state_name}","timestamp":{timestamp},"time_in_state":{time_in_state},"error_count":{error_count}}}'


def create_event_message(
    event_type: str,
    timestamp: float,
    data: Dict[str, Any],
    source: str = "voice_assistant_core",
) -> str:
    """
    Create optimized event message with minimal serialization overhead.

    Args:
        event_type: Event type name
        timestamp: Event timestamp
        data: Event data dictionary
        source: Event source identifier

    Returns:
        Compact JSON string
    """
    data_json = fast_json_dumps(data)
    return f'{{"event_type":"{event_type}","timestamp":{timestamp},"data":{data_json},"source":"{source}"}}'
