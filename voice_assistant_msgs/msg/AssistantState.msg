# Voice assistant state information
# Tracks state machine transitions for monitoring and debugging
# Published periodically and on state changes

# Current state name (e.g., "IDLE", "STREAMING_AUDIO", "TRANSCRIBING")
string current_state

# Previous state name (for transition tracking)
string previous_state

# Timestamp when state transition occurred
builtin_interfaces/Time transition_time

# Additional state data as JSON (e.g., error info, metrics)
string state_data