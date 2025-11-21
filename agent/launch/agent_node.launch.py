"""
Launch file for Agent Node.

This launch file starts the LangChain-based agent node with proper
configuration for OpenAI integration and ROS2 communication.

Author: astra
License: Apache-2.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for agent node."""

    # Find agent package share directory for config files
    agent_config_path = PathJoinSubstitution(
        [FindPackageShare("agent"), "config", "agent_config.yaml"]
    )

    # Declare launch arguments
    declare_config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=agent_config_path,
        description="Path to agent configuration YAML file",
    )

    declare_provider_arg = DeclareLaunchArgument(
        "provider", default_value="groq", description="LLM provider (openai, groq)"
    )

    declare_model_arg = DeclareLaunchArgument(
        "model",
        default_value="llama-3.3-70b-versatile",
        description="LLM model name (provider-specific)",
    )

    declare_temperature_arg = DeclareLaunchArgument(
        "temperature", default_value="0.7", description="LLM temperature (0.0-1.0)"
    )

    declare_max_tokens_arg = DeclareLaunchArgument(
        "max_tokens", default_value="500", description="Maximum tokens in LLM response"
    )

    declare_log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="ROS2 log level (debug, info, warn, error)",
    )

    declare_namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="/voice_assistant",
        description="ROS2 namespace for the agent node",
    )

    # Agent node configuration
    agent_node = Node(
        package="agent",
        executable="agent_node",
        name="agent_node",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[
            LaunchConfiguration("config_file"),
            {
                # Override config file parameters with launch arguments
                "llm.provider": LaunchConfiguration("provider"),
                "llm.model": LaunchConfiguration("model"),
                "llm.temperature": LaunchConfiguration("temperature"),
                "llm.max_tokens": LaunchConfiguration("max_tokens"),
            },
        ],
        # Topic remapping for integration with voice assistant pipeline
        remappings=[
            # Subscribe to transcription results from STT
            ("~/transcription_result", "/voice_assistant/transcription_result"),
            # Publish LLM responses for TTS
            ("~/llm_response", "/voice_assistant/llm_response"),
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
        # Respawn configuration for reliability
        respawn=True,
        respawn_delay=5.0,
    )

    return LaunchDescription(
        [
            declare_config_file_arg,
            declare_provider_arg,
            declare_model_arg,
            declare_temperature_arg,
            declare_max_tokens_arg,
            declare_log_level_arg,
            declare_namespace_arg,
            agent_node,
        ]
    )
