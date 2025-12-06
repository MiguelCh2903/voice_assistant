# Guía Rápida de Migración a VAPI

## Resumen de Cambios

Esta migración simplifica tu proyecto de asistente de voz reemplazando la implementación manual (STT, LLM, TTS separados) con VAPI, una plataforma de voz todo-en-uno.

### ✅ Lo que se ha hecho:

1. **Creado rama `vapi-migration`** en el repositorio para preservar la versión actual
2. **Eliminados módulos manuales**: `agent/`, `stt/`, `tts/`
3. **Actualizado `voice_assistant_core`** con integración VAPI
4. **Creado nuevo nodo**: `vapi_voice_assistant_node.py`
5. **Actualizado configuración**: Nuevos archivos YAML con parámetros VAPI
6. **Actualizado launch file**: `vapi_voice_assistant.launch.py`

### 🔑 Conceptos Clave:

- **Sin palabra de activación**: La llamada inicia automáticamente al lanzar el nodo
- **Streaming continuo**: El audio de ESPHome fluye directamente a VAPI
- **Procesamiento en la nube**: VAPI maneja STT, LLM y TTS internamente
- **Conexión ESPHome preservada**: Se mantiene la misma interfaz con tu dispositivo

## Pasos para Usar la Nueva Versión

### 1. Cambiar a la rama de migración

```bash
cd /home/astra/ros2_ws/src/voice
git checkout vapi-migration
```

### 2. Instalar dependencias

```bash
cd /home/astra/ros2_ws/src/voice/voice_assistant_core
pip install -r requirements.txt
```

Las nuevas dependencias principales son:
- `vapi_python>=0.1.9`: SDK oficial de VAPI
- `aioesphomeapi>=21.0.0`: Cliente ESPHome (ya lo tenías)

### 3. Configurar variables de entorno

Puedes usar el script interactivo:

```bash
./scripts/setup_vapi_env.sh
```

O crear manualmente `~/.env`:

```bash
# VAPI Configuration
export VAPI_API_KEY="tu-api-key-de-vapi"
export VAPI_ASSISTANT_ID="tu-assistant-id-de-vapi"
export VAPI_API_URL="https://api.vapi.ai"
export VAPI_AUTO_START="true"

# ESPHome Device (mantén tu configuración actual)
export ESPHOME_HOST="192.168.1.71"
export ESPHOME_PORT="6053"
export ESPHOME_PASSWORD=""
export ESPHOME_ENCRYPTION_KEY="tu-encryption-key"
```

### 4. Cargar variables de entorno

```bash
source ~/.env
```

### 5. Construir el workspace

```bash
cd /home/astra/ros2_ws
colcon build --packages-select voice_assistant_core voice_assistant_msgs
source install/setup.bash
```

### 6. Lanzar el asistente

```bash
ros2 launch voice_assistant_core vapi_voice_assistant.launch.py
```

## Configuración de VAPI

### Obtener API Key y Assistant ID

1. Ve a [vapi.ai](https://vapi.ai) y crea una cuenta
2. En el dashboard, ve a "Settings" → "API Keys" y crea una nueva key
3. Ve a "Assistants" y crea un nuevo asistente:
   - **First Message**: "Hola, ¿en qué puedo ayudarte?"
   - **System Prompt**: Instrucciones para el comportamiento del asistente
   - **Model**: Selecciona un modelo LLM (GPT-4, Claude, etc.)
   - **Voice**: Selecciona una voz para TTS
4. Copia el "Assistant ID" del asistente creado

### Ejemplo de Configuración de Asistente VAPI

```json
{
  "firstMessage": "¡Hola! Soy tu asistente de voz. ¿Cómo puedo ayudarte hoy?",
  "systemPrompt": "Eres un asistente útil y amigable. Responde de manera concisa y clara.",
  "model": "gpt-4",
  "voice": "jennifer-playht",
  "recordingEnabled": true,
  "endCallOnHangup": false
}
```

## Cómo Funciona el Streaming

### Flujo de Audio

```
1. ESPHome Device (Micrófono)
   ↓ Audio chunks (PCM 16kHz, 16-bit, mono)
   
2. ESPHomeClientWrapper
   ↓ Callback: _on_esphome_audio()
   
3. VapiClient.stream_audio()
   ↓ Queue + Background task
   
4. VAPI (vía Daily.co WebRTC)
   ↓ Procesamiento en la nube (STT → LLM → TTS)
   
5. Respuesta de vuelta a ESPHome
   ↓ Audio de la respuesta
   
6. ESPHome Device (Speaker)
```

### Inicio Automático de Llamada

Cuando `vapi.auto_start_call: true`:
- El nodo inicia una llamada VAPI al arrancar
- No necesitas palabra de activación
- El micrófono de ESPHome empieza a streamear inmediatamente
- Puedes hablar directamente al dispositivo

## Topics ROS2

### Topics Publicados

- **`/voice_assistant/assistant_state`**: Estado del asistente (idle/active)
- **`/voice_assistant/voice_event`**: Eventos de voz (speech_start, speech_end, transcript, response, error)

### Ejemplo de Suscripción

```python
# Escuchar eventos de voz
ros2 topic echo /voice_assistant/voice_event

# Ver estado del asistente
ros2 topic echo /voice_assistant/assistant_state
```

## Diferencias con la Implementación Anterior

| Aspecto | Antes (Manual) | Ahora (VAPI) |
|---------|----------------|--------------|
| **Nodos ROS2** | 4 nodos (core, stt, agent, tts) | 1 nodo (vapi_voice_assistant) |
| **STT** | API externa separada | Integrado en VAPI |
| **LLM** | API externa separada | Integrado en VAPI |
| **TTS** | API externa separada | Integrado en VAPI |
| **Palabra de activación** | Requerida | Opcional (por defecto OFF) |
| **Turn detection** | ML local (ONNX) | Manejado por VAPI |
| **VAD** | PicoVoice Cobra local | Manejado por VAPI |
| **Configuración** | Múltiples archivos | Un solo archivo YAML |
| **Latencia** | Variable (múltiples llamadas API) | Optimizada (pipeline integrado) |

## Troubleshooting

### Error: "VAPI API key not configured"

- Verifica que `VAPI_API_KEY` esté en `~/.env`
- Asegúrate de haber ejecutado `source ~/.env`
- Verifica el API key en el dashboard de VAPI

### Error: "Cannot connect to ESPHome device"

- Verifica la IP del dispositivo: `ping $ESPHOME_HOST`
- Comprueba que el puerto sea correcto (por defecto 6053)
- Verifica la encryption key en la configuración de ESPHome
- Revisa los logs del dispositivo ESPHome

### No se escucha audio / No hay respuesta

- Verifica que el asistente VAPI esté configurado correctamente
- Comprueba la configuración de voz y modelo en VAPI
- Revisa los logs: `ros2 launch voice_assistant_core vapi_voice_assistant.launch.py`
- Verifica conectividad a internet (VAPI es cloud-based)

### El audio se corta o hay latencia

- Problema de red: Verifica tu conexión a internet
- VAPI usa WebRTC (Daily.co), requiere buena conectividad
- Considera usar un servidor VAPI local si la latencia es crítica

## Volver a la Versión Anterior

Si necesitas volver a la implementación manual:

```bash
cd /home/astra/ros2_ws/src/voice
git checkout main
cd /home/astra/ros2_ws
colcon build --packages-select voice_assistant_core agent stt tts
source install/setup.bash
```

## Próximos Pasos

1. **Personalizar el asistente**: Modifica el system prompt en VAPI dashboard
2. **Agregar funciones**: VAPI soporta function calling para integrar con ROS2
3. **Optimizar audio**: Ajusta parámetros de audio en `config/development.yaml`
4. **Monitorear performance**: Habilita métricas en la configuración
5. **Pruebas de conversación**: Habla con el asistente y refina el comportamiento

## Recursos Adicionales

- **VAPI Docs**: [docs.vapi.ai](https://docs.vapi.ai)
- **VAPI Python SDK**: [github.com/VapiAI/client-sdk-python](https://github.com/VapiAI/client-sdk-python)
- **ESPHome Voice Assistant**: [esphome.io/components/voice_assistant](https://esphome.io/components/voice_assistant.html)
- **README completo**: Ver `README_VAPI.md` para documentación detallada

## Soporte

Si tienes problemas:
1. Revisa los logs de ROS2
2. Verifica la configuración de VAPI dashboard
3. Comprueba logs del dispositivo ESPHome
4. Consulta la documentación de VAPI

---

**Nota**: Esta migración simplifica significativamente el sistema eliminando la complejidad de gestionar múltiples servicios API y reduciendo la latencia con un pipeline integrado.
