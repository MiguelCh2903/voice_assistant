#!/bin/bash

# Script to setup environment variables for VAPI Voice Assistant
# This script helps configure the necessary environment variables

echo "=================================="
echo "VAPI Voice Assistant Setup"
echo "=================================="
echo ""

ENV_FILE="$HOME/.env"

# Function to prompt for variable
prompt_var() {
    local var_name=$1
    local var_desc=$2
    local var_default=$3
    local current_value=$(grep "^export $var_name=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2- | tr -d '"')
    
    if [ -n "$current_value" ]; then
        echo "Current $var_name: $current_value"
        read -p "$var_desc [$current_value]: " new_value
        if [ -z "$new_value" ]; then
            new_value="$current_value"
        fi
    else
        read -p "$var_desc [$var_default]: " new_value
        if [ -z "$new_value" ]; then
            new_value="$var_default"
        fi
    fi
    
    echo "$new_value"
}

# Create or backup existing .env file
if [ -f "$ENV_FILE" ]; then
    echo "Backing up existing .env file to .env.backup"
    cp "$ENV_FILE" "$ENV_FILE.backup"
    # Remove old VAPI and ESPHOME variables
    sed -i '/^export VAPI_/d' "$ENV_FILE"
    sed -i '/^export ESPHOME_/d' "$ENV_FILE"
else
    touch "$ENV_FILE"
fi

echo "Configuring VAPI settings..."
echo ""

# VAPI Configuration
VAPI_API_KEY=$(prompt_var "VAPI_API_KEY" "Enter your VAPI API key" "")
VAPI_ASSISTANT_ID=$(prompt_var "VAPI_ASSISTANT_ID" "Enter your VAPI Assistant ID" "")
VAPI_API_URL=$(prompt_var "VAPI_API_URL" "Enter VAPI API URL" "https://api.vapi.ai")
VAPI_AUTO_START=$(prompt_var "VAPI_AUTO_START" "Auto-start call on launch? (true/false)" "true")

echo ""
echo "Configuring ESPHome device settings..."
echo ""

# ESPHome Configuration
ESPHOME_HOST=$(prompt_var "ESPHOME_HOST" "Enter ESPHome device IP address" "192.168.1.71")
ESPHOME_PORT=$(prompt_var "ESPHOME_PORT" "Enter ESPHome API port" "6053")
ESPHOME_PASSWORD=$(prompt_var "ESPHOME_PASSWORD" "Enter ESPHome API password (leave empty if none)" "")
ESPHOME_ENCRYPTION_KEY=$(prompt_var "ESPHOME_ENCRYPTION_KEY" "Enter ESPHome encryption key" "")

# Write to .env file
echo "" >> "$ENV_FILE"
echo "# VAPI Voice Assistant Configuration" >> "$ENV_FILE"
echo "# Generated on $(date)" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"
echo "# VAPI Settings" >> "$ENV_FILE"
echo "export VAPI_API_KEY=\"$VAPI_API_KEY\"" >> "$ENV_FILE"
echo "export VAPI_ASSISTANT_ID=\"$VAPI_ASSISTANT_ID\"" >> "$ENV_FILE"
echo "export VAPI_API_URL=\"$VAPI_API_URL\"" >> "$ENV_FILE"
echo "export VAPI_AUTO_START=\"$VAPI_AUTO_START\"" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"
echo "# ESPHome Device Settings" >> "$ENV_FILE"
echo "export ESPHOME_HOST=\"$ESPHOME_HOST\"" >> "$ENV_FILE"
echo "export ESPHOME_PORT=\"$ESPHOME_PORT\"" >> "$ENV_FILE"
echo "export ESPHOME_PASSWORD=\"$ESPHOME_PASSWORD\"" >> "$ENV_FILE"
echo "export ESPHOME_ENCRYPTION_KEY=\"$ESPHOME_ENCRYPTION_KEY\"" >> "$ENV_FILE"

echo ""
echo "=================================="
echo "Configuration saved to $ENV_FILE"
echo "=================================="
echo ""
echo "To load the environment variables, run:"
echo "  source ~/.env"
echo ""
echo "To test your configuration, run:"
echo "  ros2 launch voice_assistant_core vapi_voice_assistant.launch.py"
echo ""
