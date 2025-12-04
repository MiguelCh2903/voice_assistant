# LLM processing response
# Contains AI-generated response from LangChain/OpenAI GPT-4o-mini
# Supports streaming responses with sentence-level chunking

# AI-generated response text (may be sentence chunk or full response)
string response_text

# Detected user intent (e.g., "chat", "command", "question")
string intent

# Response confidence from LLM (0.0 = uncertain, 1.0 = confident)
float32 confidence

# Whether conversation should continue (true for streaming chunks)
bool continue_conversation

# Conversation ID for context tracking across turns
string conversation_id

# Extracted entities from conversation (e.g., names, dates, locations)
string[] entities

# Key concepts or topics from response
string[] keywords