"""Launch file for VAPI voice assistant node."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for VAPI voice assistant."""

    # Auto-detect environment mode from ROS2_ENV variable
    ros_env = os.environ.get("ROS2_ENV", "development")
    config_filename = (
        "production.yaml" if ros_env == "production" else "development.yaml"
    )

    print(
        f"🤖 VAPI Voice Assistant launching in {ros_env.upper()} mode using {config_filename}"
    )

    # Auto-selected config file path
    auto_config_path = PathJoinSubstitution(
        [FindPackageShare("voice_assistant_core"), "config", config_filename]
    )

    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=auto_config_path,
        description="Path to configuration YAML file (auto-selected based on ROS2_ENV)",
    )

    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="/voice_assistant",
        description="ROS2 namespace for the node",
    )

    # VAPI voice assistant node
    vapi_node = Node(
        package="voice_assistant_core",
        executable="vapi_voice_assistant",
        name="vapi_voice_assistant",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            LaunchConfiguration("config_file"),
            {
                # Pass environment variables directly
                "vapi.api_key": os.environ.get("VAPI_API_KEY", ""),
                "vapi.api_url": os.environ.get("VAPI_API_URL", "https://api.vapi.ai"),
                "vapi.assistant_id": os.environ.get("VAPI_ASSISTANT_ID", ""),
                "vapi.auto_start_call": os.environ.get("VAPI_AUTO_START", "true") == "true",
                "device.host": os.environ.get("ESPHOME_HOST", "192.168.1.71"),
                "device.port": int(os.environ.get("ESPHOME_PORT", "6053")),
                "device.password": os.environ.get("ESPHOME_PASSWORD", ""),
                "device.encryption_key": os.environ.get("ESPHOME_ENCRYPTION_KEY", ""),
            },
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            config_file_arg,
            namespace_arg,
            vapi_node,
        ]
    )
