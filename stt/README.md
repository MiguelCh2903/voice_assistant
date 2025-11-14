# Speech-to-Text (STT) Package

Este package implementa la funcionalidad de speech-to-text para el sistema de asistente de voz usando la API de Deepgram con el modelo Nova-3.

## Características

- **Integración con Deepgram**: Utiliza el modelo Nova-3 para transcripción de alta calidad
- **Streaming en tiempo real**: Procesamiento de audio en tiempo real via WebSocket
- **Detección de fin de turno**: Procesa transcripciones completas cuando se detecta el fin del turno de habla
- **Configuración flexible**: Parámetros configurables para diferentes idiomas y modelos
- **Integración ROS2**: Comunicación completa con el sistema de asistente de voz

## Requisitos

### Variables de Entorno

Debes configurar la clave de API de Deepgram como variable de entorno:

```bash
export DEEPGRAM_API_KEY="tu_clave_de_api_aqui"
```

### Dependencias de Python

Las siguientes dependencias se instalan automáticamente con el package:

- `deepgram-sdk>=3.5.0`: SDK oficial de Deepgram
- `websockets>=10.0`: Para conexiones WebSocket
- `numpy>=1.20.0`: Procesamiento de datos de audio

## Instalación

1. Navega al workspace de ROS2:
```bash
cd /home/astra/ros2_ws
```

2. Instala las dependencias de Python:
```bash
pip install -r src/voice/stt/requirements.txt
```

3. Compila el workspace:
```bash
colcon build --packages-select stt
```

4. Source el workspace:
```bash
source install/setup.bash
```

## Uso

### Lanzamiento Básico

```bash
ros2 launch stt stt_node.launch.py
```

### Lanzamiento con Parámetros Personalizados

```bash
ros2 launch stt stt_node.launch.py \
    deepgram_model:=nova-2 \
    language:=es \
    sample_rate:=16000 \
    processing_timeout:=30.0 \
    log_level:=info
```

### Parámetros Disponibles

- `deepgram_model`: Modelo de Deepgram a utilizar (por defecto: "nova-2")
- `language`: Código de idioma para transcripción (por defecto: "es")
- `sample_rate`: Frecuencia de muestreo del audio en Hz (por defecto: 16000)
- `processing_timeout`: Timeout de procesamiento en segundos (por defecto: 30.0)
- `min_confidence`: Umbral mínimo de confianza (por defecto: 0.5)
- `log_level`: Nivel de logging (por defecto: "info")

## Tópicos ROS2

### Suscripciones

- `/voice_assistant/voice_assistant_core/audio_chunk`: Chunks de audio del núcleo del asistente
- `/voice_assistant/voice_assistant_core/voice_event`: Eventos de voz (inicio/fin de turno)

### Publicaciones

- `/voice_assistant/transcription_result`: Resultados de transcripción completa
- `/voice_assistant/stt_event`: Eventos del sistema STT

## Flujo de Funcionamiento

1. **Inicio del Stream**: El nodo STT se suscribe a los chunks de audio del voice_assistant_core
2. **Streaming a Deepgram**: Los chunks de audio se envían en tiempo real a la API de Deepgram via WebSocket
3. **Transcripción Parcial**: Se reciben transcripciones parciales durante el proceso
4. **Detección de Fin de Turno**: Cuando el voice_assistant_core detecta fin de turno, envía un evento
5. **Finalización**: Se combina toda la transcripción parcial y se publica como resultado final

## Estructura de Mensajes

### Resultado de Transcripción

```json
{
  "text": "transcripción completa del audio",
  "confidence": 0.9,
  "language": "es",
  "processing_time": 2.5,
  "audio_metadata": {
    "sample_rate": 16000,
    "channels": 1,
    "encoding": "linear16",
    "duration_seconds": 2.5
  }
}
```

### Eventos STT

```json
{
  "event_type": "TRANSCRIPTION_COMPLETE",
  "timestamp": 1699123456.789,
  "data": {
    "transcript_length": 45,
    "processing_time": 2.5,
    "audio_chunks_processed": 125
  }
}
```

## Troubleshooting

### Error: "DEEPGRAM_API_KEY environment variable not set"

Asegúrate de que la variable de entorno esté configurada:
```bash
echo $DEEPGRAM_API_KEY
```

Si no está configurada, añádela a tu `.bashrc` o `.zshrc`:
```bash
echo 'export DEEPGRAM_API_KEY="tu_clave_aqui"' >> ~/.bashrc
source ~/.bashrc
```

### Error: "Deepgram SDK not available"

Instala el SDK de Deepgram:
```bash
pip install deepgram-sdk>=3.5.0
```

### Problemas de Conexión WebSocket

1. Verifica tu conexión a internet
2. Comprueba que la clave de API sea válida
3. Revisa los logs del nodo para errores específicos:
```bash
ros2 node info /voice_assistant/stt_node
```

### Audio Sin Transcribir

1. Verifica que los chunks de audio estén llegando:
```bash
ros2 topic echo /voice_assistant/voice_assistant_core/audio_chunk
```

2. Comprueba que los eventos de fin de turno se estén enviando:
```bash
ros2 topic echo /voice_assistant/voice_assistant_core/voice_event
```

## Desarrollo

Para modificar o extender el nodo STT:

1. Los archivos principales están en `stt/stt_node.py`
2. La configuración está en `launch/stt_node.launch.py`
3. Las dependencias están definidas en `setup.py` y `requirements.txt`

### Añadir Nuevos Modelos

Para usar diferentes modelos de Deepgram, modifica el parámetro `deepgram_model` en el archivo de lanzamiento o pásalo como argumento.

### Configurar Diferentes Idiomas

Modifica el parámetro `language` para cambiar el idioma de transcripción:
- `es`: Español
- `en`: Inglés
- `fr`: Francés
- etc.

## Licencia

MIT