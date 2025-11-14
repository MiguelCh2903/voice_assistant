# Voice assistant system event
# Contains event notifications and system messages

# Event type
string event_type

# Event message
string message

# Event timestamp
builtin_interfaces/Time timestamp

# Event priority (0=info, 1=warning, 2=error)
uint8 priority

# Additional event data as JSON
string event_data