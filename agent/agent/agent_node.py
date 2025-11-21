"""
LangChain-based LLM Agent Node for CAMI Voice Assistant.

This node integrates with the ROS2 voice assistant pipeline:
- Subscribes to transcription results from STT
- Processes user input with OpenAI GPT-4o-mini via LangChain
- Streams responses in sentence chunks for real-time TTS
- Maintains conversational context with sliding window history

Author: astra
License: Apache-2.0
"""

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import List

import rclpy
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class AgentNode(Node):
    """
    LangChain-based agent node for conversational AI in voice assistant pipeline.

    Integrates with ROS2 topics:
    - Input: /voice_assistant/transcription_result (from STT)
    - Output: /voice_assistant/llm_response (to TTS)
    """

    def __init__(self):
        """Initialize the agent node."""
        super().__init__("agent_node")

        # Setup components
        self._setup_logging()
        self._setup_parameters()
        self._setup_llm()
        self._setup_conversation_state()
        self._setup_publishers()
        self._setup_subscribers()

        self._logger.info("🤖 Agent node initialized successfully")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        self._logger = self.get_logger()
        self._logger.info("Agent node starting...")

    def _setup_parameters(self) -> None:
        """Setup ROS2 parameters from config file."""
        # LLM configuration
        self.declare_parameter("llm.provider", "openai")
        self.declare_parameter("llm.model", "gpt-4o-mini")
        self.declare_parameter("llm.temperature", 0.7)
        self.declare_parameter("llm.max_tokens", 500)
        self.declare_parameter("llm.streaming", True)
        self.declare_parameter("llm.timeout", 10.0)

        # Conversation configuration
        self.declare_parameter("conversation.max_history_messages", 10)
        self.declare_parameter("conversation.id_prefix", "cami_conv_")

        # Streaming configuration
        self.declare_parameter("streaming.accumulate_sentences", True)
        self.declare_parameter("streaming.sentence_delimiters", "[.?!]")
        self.declare_parameter(
            "streaming.abbreviations",
            [
                "Dr.",
                "Dra.",
                "Sr.",
                "Sra.",
                "Ing.",
                "Lic.",
                "Prof.",
                "etc.",
                "p.ej.",
                "aprox.",
            ],
        )

        # Prompt configuration
        self.declare_parameter("prompt.instructions_file", "cami_host_instructions.txt")

        # Response metadata
        self.declare_parameter("response.default_intent", "chat")
        self.declare_parameter("response.default_confidence", 0.95)

        self._logger.info("Parameters declared")

    def _setup_llm(self) -> None:
        """Initialize LangChain chat model based on configured provider."""
        provider = self.get_parameter("llm.provider").value.lower()
        model_name = self.get_parameter("llm.model").value
        temperature = self.get_parameter("llm.temperature").value
        max_tokens = self.get_parameter("llm.max_tokens").value
        timeout = self.get_parameter("llm.timeout").value

        # Initialize based on provider
        if provider == "openai":
            self._llm = self._init_openai(model_name, temperature, max_tokens, timeout)
        elif provider == "groq":
            self._llm = self._init_groq(model_name, temperature, max_tokens, timeout)
        else:
            self._logger.error(f"❌ Unsupported provider: {provider}")
            raise ValueError(
                f"Unsupported LLM provider '{provider}'. "
                "Supported providers: 'openai', 'groq'"
            )

        self._logger.info(
            f"✅ LLM initialized: {provider}/{model_name} (temp={temperature})"
        )

        # Load system instructions
        self._load_system_instructions()

    def _init_openai(
        self, model_name: str, temperature: float, max_tokens: int, timeout: float
    ) -> ChatOpenAI:
        """Initialize OpenAI chat model."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self._logger.error(
                "❌ OPENAI_API_KEY not found in environment variables. "
                "Please set it in /home/astra/ros2_ws/.env"
            )
            raise ValueError("OPENAI_API_KEY environment variable is required")

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
            streaming=True,
        )

    def _init_groq(
        self, model_name: str, temperature: float, max_tokens: int, timeout: float
    ) -> ChatGroq:
        """Initialize Groq chat model."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            self._logger.error(
                "❌ GROQ_API_KEY not found in environment variables. "
                "Please set it in /home/astra/ros2_ws/.env"
            )
            raise ValueError("GROQ_API_KEY environment variable is required")

        return ChatGroq(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            api_key=api_key,
        )

    def _load_system_instructions(self) -> None:
        """Load system instructions from prompt file."""
        instructions_file = self.get_parameter("prompt.instructions_file").value

        # Try to find the prompt file in package share directory
        try:
            from ament_index_python.packages import get_package_share_directory

            package_share = get_package_share_directory("agent")
            prompt_path = Path(package_share) / "prompts" / instructions_file
        except Exception:
            # Fallback to relative path during development
            prompt_path = Path(__file__).parent.parent / "prompts" / instructions_file

        if not prompt_path.exists():
            self._logger.error(f"❌ Prompt file not found: {prompt_path}")
            self._system_instructions = (
                "Eres un asistente de voz para CAMI. "
                "Responde de forma concisa y profesional."
            )
        else:
            with open(prompt_path, "r", encoding="utf-8") as f:
                self._system_instructions = f.read()
            self._logger.info(f"✅ System instructions loaded from {prompt_path}")

    def _setup_conversation_state(self) -> None:
        """Initialize conversation state and history management."""
        # Conversation history (list of messages)
        self._message_history: List[SystemMessage | HumanMessage | AIMessage] = []

        # Current conversation ID
        id_prefix = self.get_parameter("conversation.id_prefix").value
        self._conversation_id = f"{id_prefix}{uuid.uuid4().hex[:8]}"

        # Streaming buffer for accumulating sentence chunks
        self._streaming_buffer = ""

        # Compile abbreviations regex for sentence detection
        abbreviations = self.get_parameter("streaming.abbreviations").value
        # Escape special regex characters in abbreviations
        escaped_abbrevs = [re.escape(abbr) for abbr in abbreviations]
        self._abbreviations_pattern = re.compile(
            "|".join(escaped_abbrevs), re.IGNORECASE
        )

        self._logger.info(f"Conversation initialized: {self._conversation_id}")

    def _setup_publishers(self) -> None:
        """Setup ROS2 publishers."""
        # QoS profile for reliable event delivery
        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Publisher for LLM responses
        self._llm_response_pub = self.create_publisher(
            String,
            "/voice_assistant/llm_response",
            event_qos,
        )

        self._logger.info("Publishers created")

    def _setup_subscribers(self) -> None:
        """Setup ROS2 subscribers."""
        # QoS profile for reliable event delivery
        event_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Subscriber for transcription results
        self._transcription_sub = self.create_subscription(
            String,
            "/voice_assistant/transcription_result",
            self._transcription_callback,
            event_qos,
        )

        self._logger.info("Subscribers created")

    def _transcription_callback(self, msg: String) -> None:
        """
        Handle incoming transcription from STT.

        Args:
            msg: String message containing JSON-encoded transcription
        """
        try:
            # Parse transcription JSON
            data = json.loads(msg.data)
            user_text = data.get("text", "")
            confidence = data.get("confidence", 0.0)
            is_final = data.get(
                "is_final", True
            )  # Default to True for backwards compatibility

            # OPTIMIZATION: Only process final transcriptions to avoid processing intermediate results
            if not is_final:
                self._logger.debug(
                    f"Skipping incremental transcription: '{user_text[:30]}...'"
                )
                return

            if not user_text.strip():
                self._logger.warning("Received empty transcription")
                return

            self._logger.info(
                f"📝 Final transcription received: '{user_text}' (conf={confidence:.2f})"
            )

            # Process with LLM immediately
            self._process_user_input(user_text)

        except json.JSONDecodeError as e:
            self._logger.error(f"Invalid transcription JSON: {e}")
        except Exception as e:
            self._logger.error(f"Error processing transcription: {e}")

    def _process_user_input(self, user_text: str) -> None:
        """
        Process user input with LLM and stream response.

        Args:
            user_text: User's transcribed message
        """
        start_time = time.time()

        # Add user message to history
        self._add_to_history(HumanMessage(content=user_text))

        # Prepare messages for LLM
        messages = self._build_message_list()

        # Stream LLM response
        try:
            self._logger.info("🤖 Generating LLM response...")
            self._stream_llm_response(messages)

            processing_time = time.time() - start_time
            self._logger.info(f"✅ Response completed in {processing_time:.2f}s")

        except Exception as e:
            self._logger.error(f"❌ LLM processing error: {e}")
            self._publish_error_response(str(e))

    def _build_message_list(self) -> List[SystemMessage | HumanMessage | AIMessage]:
        """
        Build message list for LLM with system prompt and history.

        Returns:
            List of messages for LLM
        """
        messages = [SystemMessage(content=self._system_instructions)]
        messages.extend(self._message_history)
        return messages

    def _stream_llm_response(self, messages: List) -> None:
        """
        Stream LLM response and publish sentence chunks in real-time.

        Args:
            messages: List of messages for LLM
        """
        self._streaming_buffer = ""
        full_response = ""

        # Stream tokens from LLM using astream_events
        for chunk in self._llm.stream(messages):
            # Extract token content
            if hasattr(chunk, "content") and chunk.content:
                token = chunk.content
                self._streaming_buffer += token
                full_response += token

                # Check for complete sentences
                self._check_and_publish_sentences()

        # Publish any remaining content
        if self._streaming_buffer.strip():
            self._publish_sentence(
                self._streaming_buffer.strip(), continue_conversation=False
            )
            self._streaming_buffer = ""

        # Add assistant response to history
        self._add_to_history(AIMessage(content=full_response))

    def _check_and_publish_sentences(self) -> None:
        """
        Check buffer for complete sentences and publish them.
        Uses regex to detect sentence boundaries while avoiding abbreviations.
        """
        if not self.get_parameter("streaming.accumulate_sentences").value:
            return

        delimiter_pattern = self.get_parameter("streaming.sentence_delimiters").value

        # Find sentence boundaries
        # Pattern: sentence delimiter followed by space or end of string
        sentence_pattern = f"{delimiter_pattern}(?:\\s+|$)"

        while True:
            match = re.search(sentence_pattern, self._streaming_buffer)
            if not match:
                break

            # Extract potential sentence
            sentence_end = match.end()
            potential_sentence = self._streaming_buffer[:sentence_end].strip()

            # Check if it's an abbreviation (false positive)
            if self._is_abbreviation_boundary(potential_sentence):
                # Not a real sentence boundary, continue accumulating
                break

            # Valid sentence found - publish it
            if potential_sentence:
                self._publish_sentence(potential_sentence, continue_conversation=True)

            # Remove published sentence from buffer
            self._streaming_buffer = self._streaming_buffer[sentence_end:].lstrip()

    def _is_abbreviation_boundary(self, text: str) -> bool:
        """
        Check if text ends with a known abbreviation (false positive for sentence end).

        Args:
            text: Text to check

        Returns:
            True if ends with abbreviation
        """
        if self._abbreviations_pattern.search(text):
            # Check if abbreviation is at the end
            for abbr in self.get_parameter("streaming.abbreviations").value:
                if text.rstrip().endswith(abbr):
                    return True
        return False

    def _publish_sentence(self, sentence: str, continue_conversation: bool) -> None:
        """
        Publish a sentence chunk as LLM response.

        Args:
            sentence: Sentence text to publish
            continue_conversation: Whether more content is coming
        """
        default_intent = self.get_parameter("response.default_intent").value
        default_confidence = self.get_parameter("response.default_confidence").value

        response_data = {
            "response_text": sentence,
            "intent": default_intent,
            "confidence": default_confidence,
            "continue_conversation": continue_conversation,
            "conversation_id": self._conversation_id,
            "entities": [],
            "keywords": [],
        }

        msg = String()
        msg.data = json.dumps(response_data)
        self._llm_response_pub.publish(msg)

        continue_marker = "..." if continue_conversation else "✓"
        self._logger.info(f"📤 Published: '{sentence[:50]}...' [{continue_marker}]")

    def _publish_error_response(self, error_message: str) -> None:
        """
        Publish error response to LLM response topic.

        Args:
            error_message: Error description
        """
        response_data = {
            "response_text": "Lo siento, hubo un error al procesar tu solicitud.",
            "intent": "error",
            "confidence": 0.0,
            "continue_conversation": False,
            "conversation_id": self._conversation_id,
            "entities": [],
            "keywords": [],
            "error_message": error_message,
        }

        msg = String()
        msg.data = json.dumps(response_data)
        self._llm_response_pub.publish(msg)

    def _add_to_history(self, message: HumanMessage | AIMessage) -> None:
        """
        Add message to conversation history with sliding window.

        Args:
            message: Message to add
        """
        self._message_history.append(message)

        # Maintain sliding window (keep last N messages)
        max_messages = self.get_parameter("conversation.max_history_messages").value

        # Each turn has 2 messages (human + AI), so max_messages * 2 total
        max_total_messages = max_messages * 2

        if len(self._message_history) > max_total_messages:
            # Remove oldest messages (keep most recent)
            self._message_history = self._message_history[-max_total_messages:]
            self._logger.debug(
                f"History trimmed to {len(self._message_history)} messages "
                f"(max {max_messages} turns)"
            )


def main(args=None):
    """Main entry point for agent node."""
    rclpy.init(args=args)

    try:
        node = AgentNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in agent node: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
