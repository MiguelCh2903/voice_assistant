# TTS audio data for playback
# Contains synthesized speech from ElevenLabs API
# Optimized for direct playback on ESPHome devices

# PCM audio data as byte array (int16 samples)
uint8[] audio_data

# Audio format information (must match ESPHome expectations)
string format           # e.g., "pcm_s16le"
uint32 sample_rate      # Samples per second (typically 16000 Hz)
uint8 channels          # Number of audio channels (1=mono)
uint8 sample_width      # Bits per sample (typically 16)

# Synthesis metadata
string source_text      # Original text that was synthesized
string engine           # TTS engine used (e.g., "elevenlabs")

# Conversation context (for tracking and debugging)
string conversation_id