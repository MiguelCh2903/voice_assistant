from setuptools import find_packages, setup

package_name = "stt"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/stt_node.launch.py"]),
    ],
    install_requires=[
        "setuptools",
        "deepgram-sdk>=3.5.0",
        "websockets>=10.0",
        "asyncio",
        "numpy",
    ],
    zip_safe=True,
    maintainer="astra",
    maintainer_email="astra@todo.todo",
    description="Speech-to-Text package for voice assistant using Deepgram Nova-3",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "stt_node = stt.stt_node:main",
        ],
    },
)
