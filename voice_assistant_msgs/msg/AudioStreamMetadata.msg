# Audio stream metadata
# Contains technical information and statistics about audio streams
# Used for debugging and stream quality monitoring

# Stream identification
uint32 stream_id        # Unique stream identifier

# Audio format details (matches AudioChunk format)
string format           # e.g., "pcm_s16le"
uint32 sample_rate      # Samples per second
uint8 channels          # Number of audio channels
uint8 sample_width      # Bits per sample

# Stream statistics (accumulated during stream lifetime)
uint32 total_chunks     # Total number of chunks received
uint64 total_bytes      # Total bytes of audio data
float32 duration        # Stream duration in seconds

# Stream timing (for latency analysis)
builtin_interfaces/Time start_time
builtin_interfaces/Time end_time