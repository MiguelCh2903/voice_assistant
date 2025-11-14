# Audio stream metadata
# Contains technical information about audio stream

# Stream identifier
uint32 stream_id

# Audio format details
string format
uint32 sample_rate
uint8 channels
uint8 sample_width

# Stream statistics
uint32 total_chunks
uint64 total_bytes
float32 duration

# Stream timestamps
builtin_interfaces/Time start_time
builtin_interfaces/Time end_time