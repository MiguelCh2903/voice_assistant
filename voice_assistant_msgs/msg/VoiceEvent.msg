# Voice assistant system event
# Contains event notifications for monitoring and debugging
# Used for logging, telemetry, and inter-node communication

# Priority level constants
uint8 PRIORITY_INFO = 0
uint8 PRIORITY_WARNING = 1
uint8 PRIORITY_ERROR = 2

# Event type identifier (e.g., "AUDIO_STREAM_START", "ERROR_OCCURRED")
string event_type

# Human-readable event message
string message

# Timestamp when event occurred
builtin_interfaces/Time timestamp

# Event priority level (use constants above)
uint8 priority

# Additional event data as JSON (for detailed diagnostics)
string event_data