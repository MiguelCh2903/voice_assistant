# Speech-to-text transcription result
# Contains transcribed text from Deepgram Nova-3 model
# Optimized for low-latency streaming and accurate confidence metrics

# Transcribed text (may be partial or final)
string text

# Confidence score from STT model (0.0 = low, 1.0 = high)
float32 confidence

# Language detected or used (e.g., "es-419", "en-US")
string language

# Processing time in seconds (from audio start to transcription)
float32 processing_time

# Source audio metadata (for quality tracking)
AudioStreamMetadata audio_metadata