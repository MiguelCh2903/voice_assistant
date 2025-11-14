# LLM processing response
# Contains AI-generated response and conversation context

# AI response text
string response_text

# Detected intent
string intent

# Response confidence (0.0 to 1.0)
float32 confidence

# Whether to continue conversation
bool continue_conversation

# Conversation ID for context
string conversation_id

# Additional metadata
string[] entities
string[] keywords