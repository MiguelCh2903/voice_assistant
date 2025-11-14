# Audio chunk message for streaming audio data
# Contains raw audio data and metadata

# Raw audio data as byte array
uint8[] data

# Timestamp when chunk was captured
builtin_interfaces/Time timestamp

# Sequential chunk identifier within stream
uint32 sequence_id

# Audio format information
string format
uint32 sample_rate
uint8 channels
uint8 sample_width

# Stream control
bool is_final
uint32 stream_id