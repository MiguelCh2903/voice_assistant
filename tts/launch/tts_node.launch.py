"""
Launch file for TTS (Text-to-Speech) node.

This launch file starts the TTS node with proper configuration for
Eleven Labs integration and ROS2 communication.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for TTS node."""

    # Declare launch arguments
    declare_voice_id_arg = DeclareLaunchArgument(
        "voice_id",
        default_value="JBFqnCBsd6RMkjVDRZzb",
        description="Eleven Labs voice ID to use for synthesis",
    )

    declare_model_id_arg = DeclareLaunchArgument(
        "model_id",
        default_value="eleven_multilingual_v2",
        description="Eleven Labs model ID (eleven_multilingual_v2, eleven_turbo_v2, etc.)",
    )

    declare_output_format_arg = DeclareLaunchArgument(
        "output_format",
        default_value="pcm_16000",
        description="Audio output format (pcm_16000 for 16kHz PCM)",
    )

    declare_sample_rate_arg = DeclareLaunchArgument(
        "sample_rate", default_value="16000", description="Audio sample rate in Hz"
    )

    declare_log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Logging level (debug, info, warn, error)",
    )

    declare_playback_enabled_arg = DeclareLaunchArgument(
        "playback_enabled",
        default_value="true",
        description="Enable local audio playback on Bluetooth speaker",
    )

    declare_audio_device_arg = DeclareLaunchArgument(
        "audio_device",
        default_value="-1",
        description="Audio output device ID (-1 for default)",
    )

    # TTS Node
    tts_node = Node(
        package="tts",
        executable="tts_node",
        name="tts_node",
        namespace="voice_assistant",
        output="screen",
        parameters=[
            {
                # Eleven Labs configuration
                "elevenlabs.voice_id": LaunchConfiguration("voice_id"),
                "elevenlabs.model_id": LaunchConfiguration("model_id"),
                "elevenlabs.output_format": LaunchConfiguration("output_format"),
                # Audio configuration
                "audio.sample_rate": LaunchConfiguration("sample_rate"),
                "audio.channels": 1,
                "audio.sample_width": 16,
                # Processing configuration
                "processing.stream_chunk_size": 4096,
                # Playback configuration
                "playback.enabled": LaunchConfiguration("playback_enabled"),
                "playback.device_id": LaunchConfiguration("audio_device"),
            }
        ],
        remappings=[
            # Remap topics to match voice_assistant_core namespace
            ("~/tts_audio", "/voice_assistant/tts_audio"),
            ("~/tts_event", "/voice_assistant/tts_event"),
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    return LaunchDescription(
        [
            declare_voice_id_arg,
            declare_model_id_arg,
            declare_output_format_arg,
            declare_sample_rate_arg,
            declare_log_level_arg,
            declare_playback_enabled_arg,
            declare_audio_device_arg,
            tts_node,
        ]
    )
