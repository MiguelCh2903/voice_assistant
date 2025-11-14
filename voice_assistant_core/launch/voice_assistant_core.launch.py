"""Launch file for voice assistant core node."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for voice assistant core."""

    # Auto-detect environment mode from ROS2_ENV variable
    ros_env = os.environ.get("ROS2_ENV", "development")
    config_filename = (
        "production.yaml" if ros_env == "production" else "development.yaml"
    )

    print(
        f"🤖 Voice Assistant launching in {ros_env.upper()} mode using {config_filename}"
    )

    # Auto-selected config file path
    auto_config_path = PathJoinSubstitution(
        [FindPackageShare("voice_assistant_core"), "config", config_filename]
    )

    # Declare launch arguments (can override auto-detection)
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=auto_config_path,
        description="Path to configuration YAML file (auto-selected based on ROS2_ENV)",
    )

    use_dev_config_arg = DeclareLaunchArgument(
        "use_dev_config",
        default_value="false",
        description="Use development configuration",
    )

    device_host_arg = DeclareLaunchArgument(
        "device_host",
        default_value="192.168.1.71",
        description="ESPHome device IP address",
    )

    device_port_arg = DeclareLaunchArgument(
        "device_port", default_value="6053", description="ESPHome device port"
    )

    device_password_arg = DeclareLaunchArgument(
        "device_password",
        default_value="",
        description="ESPHome device password (if required)",
    )

    device_encryption_key_arg = DeclareLaunchArgument(
        "device_encryption_key",
        default_value="G10Tgcs5WRoXRkUYi7Ehtm3tGzdyPgUCRR7LyARUEDM=",
        description="ESPHome device encryption key",
    )

    device_name_arg = DeclareLaunchArgument(
        "device_name",
        default_value="Voice Assistant Device",
        description="Human-readable device name",
    )

    debug_logging_arg = DeclareLaunchArgument(
        "debug_logging", default_value="true", description="Enable debug logging"
    )

    auto_reconnect_arg = DeclareLaunchArgument(
        "auto_reconnect",
        default_value="true",
        description="Enable automatic reconnection",
    )

    namespace_arg = DeclareLaunchArgument(
        "namespace",
        default_value="/voice_assistant",
        description="ROS2 namespace for the node",
    )

    # Voice assistant core node
    voice_assistant_node = Node(
        package="voice_assistant_core",
        executable="voice_assistant_core",
        name="voice_assistant_core",
        namespace=LaunchConfiguration("namespace"),
        parameters=[
            LaunchConfiguration("config_file"),
            {
                # Pass environment variables directly (workaround for ROS2 Jazzy $(env) bug)
                "vad.access_key": os.environ.get("PICOVOICE_ACCESS_KEY", ""),
                "device.host": os.environ.get("ESPHOME_HOST", "192.168.1.71"),
                "device.port": int(os.environ.get("ESPHOME_PORT", "6053")),
                "device.password": os.environ.get("ESPHOME_PASSWORD", ""),
                "device.encryption_key": os.environ.get("ESPHOME_ENCRYPTION_KEY", ""),
                "device.name": LaunchConfiguration("device_name"),
                "debug.enable_logging": LaunchConfiguration("debug_logging"),
                "connection.auto_reconnect": LaunchConfiguration("auto_reconnect"),
            },
        ],
        output="screen",
        emulate_tty=True,
        respawn=True,  # Automatically restart if node crashes
        respawn_delay=5.0,  # Wait 5 seconds before restarting
    )

    return LaunchDescription(
        [
            config_file_arg,
            use_dev_config_arg,
            device_host_arg,
            device_port_arg,
            device_password_arg,
            device_encryption_key_arg,
            device_name_arg,
            debug_logging_arg,
            auto_reconnect_arg,
            namespace_arg,
            voice_assistant_node,
        ]
    )
