# TTS audio data for playback
# Contains synthesized speech audio

# Audio data
uint8[] audio_data

# Audio format information
string format
uint32 sample_rate
uint8 channels
uint8 sample_width

# Text that was synthesized
string source_text

# TTS engine used
string engine

# Conversation context
string conversation_id