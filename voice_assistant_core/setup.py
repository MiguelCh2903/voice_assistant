from setuptools import find_packages, setup
from glob import glob

package_name = "voice_assistant_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/scripts", glob("scripts/*.sh")),
    ],
    install_requires=[
        "setuptools",
        "aioesphomeapi>=20.0.0",
        "vapi_python>=0.1.9",
    ],
    zip_safe=True,
    maintainer="astra",
    maintainer_email="astra@todo.todo",
    description="VAPI voice assistant integration for ROS2 with ESPHome",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vapi_voice_assistant = voice_assistant_core.vapi_voice_assistant_node:main",
        ],
    },
)
