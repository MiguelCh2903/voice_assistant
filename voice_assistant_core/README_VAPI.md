# Voice Assistant Core - VAPI Integration

Simplified ROS2 voice assistant using VAPI for end-to-end voice processing with ESPHome audio streaming.

## Overview

This package provides a streamlined voice assistant that integrates:
- **VAPI**: End-to-end voice AI platform (STT, LLM, TTS all in one)
- **ESPHome**: Audio streaming from ESP32-based devices
- **ROS2**: Event publishing and system integration

Unlike the previous manual implementation, VAPI handles all speech processing internally, eliminating the need for separate STT, LLM, and TTS nodes.

## Architecture

```
┌─────────────────┐
│  ESPHome Device │ (Microphone + Speaker)
│   (ESP32-S3)    │
└────────┬────────┘
         │ Audio Stream (aioesphomeapi)
         ▼
┌─────────────────────────┐
│ VAPI Voice Assistant    │
│       Node              │
│  ┌──────────────────┐   │
│  │  ESPHome Client  │   │
│  └─────────┬────────┘   │
│            │             │
│  ┌─────────▼────────┐   │
│  │   VAPI Client    │   │ ──► ROS2 Topics
│  │  (STT+LLM+TTS)   │   │     - /voice_event
│  └──────────────────┘   │     - /assistant_state
└─────────────────────────┘
         │
         ▼ WebRTC/Daily.co
┌─────────────────┐
│   VAPI Cloud    │
│   API Service   │
└─────────────────┘
```

## Features

- **No Wake Word Required**: Call starts automatically when node launches
- **Continuous Audio Streaming**: Audio flows from ESPHome directly to VAPI
- **Zero-Configuration Speech Processing**: VAPI handles STT, LLM, and TTS
- **ROS2 Event Publishing**: Voice events published for integration with other nodes
- **Automatic Reconnection**: Handles ESPHome and VAPI connection failures

## Installation

### Prerequisites

1. **Python Dependencies**:
```bash
cd /home/astra/ros2_ws/src/voice/voice_assistant_core
pip install -r requirements.txt
```

2. **Environment Variables** (create `~/.env`):
```bash
# VAPI Configuration
export VAPI_API_KEY="your-vapi-api-key"
export VAPI_ASSISTANT_ID="your-vapi-assistant-id"
export VAPI_AUTO_START="true"

# ESPHome Device Configuration
export ESPHOME_HOST="192.168.1.71"
export ESPHOME_PORT="6053"
export ESPHOME_PASSWORD=""
export ESPHOME_ENCRYPTION_KEY="your-encryption-key"
```

3. **Load environment variables**:
```bash
source ~/.env
```

### Build

```bash
cd /home/astra/ros2_ws
colcon build --packages-select voice_assistant_core voice_assistant_msgs
source install/setup.bash
```

## Usage

### Launch VAPI Voice Assistant

```bash
# Using development configuration
ros2 launch voice_assistant_core vapi_voice_assistant.launch.py

# Using production configuration
ROS2_ENV=production ros2 launch voice_assistant_core vapi_voice_assistant.launch.py
```

### Launch with Custom Parameters

```bash
ros2 launch voice_assistant_core vapi_voice_assistant.launch.py \
  namespace:=/my_assistant \
  config_file:=/path/to/custom_config.yaml
```

## Configuration

Configuration is managed through YAML files in the `config/` directory:

- `development.yaml`: Development/testing configuration
- `production.yaml`: Production configuration

### Key Configuration Parameters

```yaml
voice_assistant_core:
  ros__parameters:
    # VAPI Settings
    vapi:
      api_key: "$(env VAPI_API_KEY)"
      api_url: "https://api.vapi.ai"
      assistant_id: "$(env VAPI_ASSISTANT_ID)"
      auto_start_call: true
      
    # ESPHome Device
    device:
      host: "$(env ESPHOME_HOST)"
      port: 6053
      password: "$(env ESPHOME_PASSWORD)"
      encryption_key: "$(env ESPHOME_ENCRYPTION_KEY)"
      
    # Audio Settings
    audio:
      sample_rate: 16000
      sample_width: 16
      channels: 1
```

## ROS2 Topics

### Published Topics

- **`/voice_assistant/assistant_state`** (`voice_assistant_msgs/AssistantState`)
  - Current state of the assistant (idle/active)
  - Published at 1 Hz

- **`/voice_assistant/voice_event`** (`voice_assistant_msgs/VoiceEvent`)
  - Voice events (speech_start, speech_end, transcript, response, error)
  - Published on event occurrence

## VAPI Setup

1. Create a VAPI account at [vapi.ai](https://vapi.ai)
2. Create an assistant in the VAPI dashboard
3. Configure your assistant with:
   - First message (greeting)
   - System prompt (instructions)
   - Voice selection
   - LLM model selection
4. Copy your assistant ID and API key
5. Add them to your `~/.env` file

## ESPHome Device Setup

Your ESPHome device must support the Voice Assistant feature:

```yaml
# Example ESPHome configuration
esphome:
  name: voice-assistant
  
esp32:
  board: esp32-s3-devkitc-1
  
api:
  encryption:
    key: "your-encryption-key"
    
voice_assistant:
  microphone: mic_id
  speaker: speaker_id
```

## Troubleshooting

### VAPI Connection Issues

- Verify `VAPI_API_KEY` and `VAPI_ASSISTANT_ID` are set correctly
- Check VAPI dashboard for API key status
- Ensure internet connectivity

### ESPHome Connection Issues

- Verify device IP address and port
- Check encryption key matches ESPHome config
- Ensure device is on the same network
- Check ESPHome device logs

### No Audio Streaming

- Verify ESPHome device microphone is working
- Check audio format (16kHz, 16-bit, mono PCM)
- Monitor ROS2 logs for streaming errors

### Debug Logging

Enable debug logging in `config/development.yaml`:

```yaml
debug:
  log_level: "DEBUG"
```

## Migration from Manual Implementation

If migrating from the previous manual implementation:

1. **Remove old packages**: The `agent`, `stt`, and `tts` packages are no longer needed
2. **Update configuration**: Use the new VAPI configuration format
3. **Update launch files**: Use `vapi_voice_assistant.launch.py`
4. **Set environment variables**: Add VAPI credentials to `~/.env`

## Development

### Project Structure

```
voice_assistant_core/
├── config/
│   ├── development.yaml
│   └── production.yaml
├── launch/
│   ├── vapi_voice_assistant.launch.py
│   └── voice_assistant_core.launch.py (legacy)
├── voice_assistant_core/
│   ├── vapi/
│   │   ├── __init__.py
│   │   └── vapi_client.py
│   ├── communication/
│   │   └── esphome_client.py
│   ├── vapi_voice_assistant_node.py
│   └── voice_assistant_node.py (legacy)
├── requirements.txt
└── setup.py
```

### Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=voice_assistant_core
```

## License

Apache-2.0

## Credits

- **VAPI**: [vapi.ai](https://vapi.ai)
- **ESPHome**: [esphome.io](https://esphome.io)
- **aioesphomeapi**: [github.com/esphome/aioesphome](https://github.com/esphome/aioesphome)
