"""
Launch file for STT (Speech-to-Text) node.

This launch file starts the STT node with proper configuration for
Deepgram Nova-3 integration and ROS2 communication.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for STT node."""

    # Declare launch arguments
    declare_deepgram_model_arg = DeclareLaunchArgument(
        "deepgram_model",
        default_value="nova-3",
        description="Deepgram model to use (nova-3 supports Spanish natively, nova-2 for other languages)",
    )

    declare_language_arg = DeclareLaunchArgument(
        "language",
        default_value="es",
        description="Language code for transcription (es, en, etc.)",
    )

    declare_sample_rate_arg = DeclareLaunchArgument(
        "sample_rate", default_value="16000", description="Audio sample rate in Hz"
    )

    declare_timeout_arg = DeclareLaunchArgument(
        "processing_timeout",
        default_value="30.0",
        description="Processing timeout in seconds",
    )

    declare_min_confidence_arg = DeclareLaunchArgument(
        "min_confidence",
        default_value="0.5",
        description="Minimum confidence threshold for transcription",
    )

    declare_log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Logging level (debug, info, warn, error)",
    )

    # STT Node
    stt_node = Node(
        package="stt",
        executable="stt_node",
        name="stt_node",
        namespace="voice_assistant",
        output="screen",
        parameters=[
            {
                # Deepgram configuration
                "deepgram.model": LaunchConfiguration("deepgram_model"),
                "deepgram.language": LaunchConfiguration("language"),
                "deepgram.smart_format": True,
                "deepgram.punctuate": True,
                "deepgram.interim_results": True,
                # Audio configuration
                "audio.sample_rate": LaunchConfiguration("sample_rate"),
                "audio.channels": 1,
                "audio.encoding": "linear16",
                # Processing configuration
                "processing.timeout_seconds": LaunchConfiguration("processing_timeout"),
                "processing.silence_timeout_seconds": 3.0,
                "processing.min_confidence": LaunchConfiguration("min_confidence"),
            }
        ],
        remappings=[
            # Remap topics to match voice_assistant_core namespace
            ("~/transcription_result", "/voice_assistant/transcription_result"),
            ("~/stt_event", "/voice_assistant/stt_event"),
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    return LaunchDescription(
        [
            declare_deepgram_model_arg,
            declare_language_arg,
            declare_sample_rate_arg,
            declare_timeout_arg,
            declare_min_confidence_arg,
            declare_log_level_arg,
            stt_node,
        ]
    )
