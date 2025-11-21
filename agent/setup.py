from setuptools import find_packages, setup
import os
from glob import glob

package_name = "agent"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
        (
            os.path.join("share", package_name, "prompts"),
            glob(os.path.join("prompts", "*.txt")),
        ),
    ],
    install_requires=[
        "setuptools",
        "langchain>=0.3.0",
        "langchain-openai>=0.2.0",
        "langchain-groq>=0.2.0",
        "langchain-core>=0.3.0",
    ],
    zip_safe=True,
    maintainer="astra",
    maintainer_email="miguel.chumacero.b@gmail.com",
    description="LangChain-based LLM agent for CAMI voice assistant",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "agent_node = agent.agent_node:main",
        ],
    },
)
