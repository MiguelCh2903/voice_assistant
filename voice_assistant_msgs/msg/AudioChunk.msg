# Audio chunk message for streaming audio data
# Optimized for low-latency audio streaming from ESPHome to STT
# Uses direct byte array (no encoding) for maximum performance

# Raw PCM audio data as byte array (int16 samples)
uint8[] data

# Timestamp when chunk was captured
builtin_interfaces/Time timestamp

# Sequential chunk identifier within stream (for ordering and duplicate detection)
uint32 sequence_id

# Audio format information
string format           # e.g., "pcm_s16le"
uint32 sample_rate      # Samples per second (typically 16000 Hz)
uint8 channels          # Number of audio channels (1=mono, 2=stereo)
uint8 sample_width      # Bits per sample (typically 16)

# Stream control flags
bool is_final           # True if this is the last chunk in stream
uint32 stream_id        # Stream identifier for grouping chunks