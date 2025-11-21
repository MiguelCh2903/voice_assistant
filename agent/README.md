# Agent Package - LangChain LLM Agent for CAMI Voice Assistant

Este paquete implementa un agente conversacional basado en LangChain para el sistema de asistente de voz CAMI. Soporta múltiples proveedores de LLM incluyendo **OpenAI** y **Groq** para generar respuestas naturales y contextuales en español.

## Características

- 🤖 **Agente Conversacional**: Utiliza LangChain con múltiples proveedores LLM
- ⚡ **Soporte Multi-Proveedor**: OpenAI (GPT-4o, GPT-4o-mini) y Groq (Llama 3.3, Mixtral)
- 🎭 **Personalidad CAMI**: Anfitrión carismático del Centro Avanzado de Mecatrónica Inteligente
- 💬 **Historial Conversacional**: Mantiene contexto con ventana deslizante de 10 mensajes
- 📡 **Streaming en Tiempo Real**: Respuestas por oraciones para baja latencia
- 🔄 **Integración Pipeline**: Entre STT y TTS en el flujo del asistente de voz
- 🛡️ **Manejo de Abreviaciones**: Detección inteligente de fin de oración en español

## Proveedores Soportados

### OpenAI
- **Modelos**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **Velocidad**: Media-Alta
- **Costo**: Medio
- **Requiere**: `OPENAI_API_KEY`

### Groq
- **Modelos**: `llama-3.3-70b-versatile`, `llama-3.1-70b-versatile`, `mixtral-8x7b-32768`
- **Velocidad**: Ultra-rápida (LPU™)
- **Costo**: Bajo (free tier generoso)
- **Requiere**: `GROQ_API_KEY`
- **Recomendado**: ⚡ Para desarrollo y producción de bajo costo

## Arquitectura

### Pipeline de Integración

```
STT Node → Transcription → Agent Node → LLM Response → TTS Node
           (JSON)                       (JSON Stream)
```

### Topics ROS2

**Suscripciones (Input)**:
- `/voice_assistant/transcription_result` - Transcripciones del STT (String con JSON)

**Publicaciones (Output)**:
- `/voice_assistant/llm_response` - Respuestas del LLM (String con JSON)

### Formato de Mensajes

**Input (Transcription)**:
```json
{
  "text": "¿qué proyectos tiene CAMI?",
  "confidence": 0.92,
  "language": "es",
  "processing_time": 1.5
}
```

**Output (LLM Response)**:
```json
{
  "response_text": "CAMI desarrolla proyectos en tres áreas principales.",
  "intent": "chat",
  "confidence": 0.95,
  "continue_conversation": true,
  "conversation_id": "cami_conv_a1b2c3d4",
  "entities": [],
  "keywords": []
}
```

## Requisitos

### Variables de Entorno

Configura las siguientes variables en `/home/astra/ros2_ws/.env`:

```bash
# Groq API Key (Recomendado - Ultra rápido y free tier generoso)
GROQ_API_KEY="gsk_..."

# OpenAI API Key (Opcional)
OPENAI_API_KEY="sk-..."

# Opcional (para debugging con LangSmith)
LANGCHAIN_API_KEY="lsv2_..."
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_PROJECT="cami-voice-assistant"
```

**Nota**: Solo necesitas la API key del proveedor que uses. Por defecto, el sistema usa **Groq**.

### Dependencias de Python

Las siguientes dependencias se instalan automáticamente:

- `langchain>=0.3.0` - Framework principal
- `langchain-openai>=0.2.0` - Integración con OpenAI
- `langchain-groq>=0.2.0` - Integración con Groq
- `langchain-core>=0.3.0` - Componentes core de LangChain

## Instalación

1. Obtén tu API key del proveedor que prefieras:
   - **Groq** (Recomendado): https://console.groq.com/
   - **OpenAI**: https://platform.openai.com/api-keys

2. Configura el archivo `.env`:
```bash
cd /home/astra/ros2_ws
nano .env  # Añade GROQ_API_KEY o OPENAI_API_KEY
```

2. Instala las dependencias de Python:
```bash
cd /home/astra/ros2_ws/src/voice/agent
pip install -r ../../requirements.txt  # Si existe
# O las dependencias se instalarán automáticamente con colcon
```

3. Compila el workspace:
```bash
cd /home/astra/ros2_ws
colcon build --packages-select agent
```

4. Source el workspace:
```bash
source install/setup.bash
```

## Uso

### Lanzamiento Básico (Groq - Default)

```bash
ros2 launch agent agent_node.launch.py
```

### Lanzamiento con OpenAI

```bash
ros2 launch agent agent_node.launch.py \
    provider:=openai \
    model:=gpt-4o-mini
```

### Lanzamiento con Groq (explícito)

```bash
ros2 launch agent agent_node.launch.py \
    provider:=groq \
    model:=llama-3.3-70b-versatile
```

### Lanzamiento con Parámetros Personalizados

```bash
ros2 launch agent agent_node.launch.py \
    provider:=groq \
    model:=mixtral-8x7b-32768 \
    temperature:=0.8 \
    max_tokens:=300 \
    log_level:=info
```

### Verificar Estado del Nodo

```bash
# Ver nodos activos
ros2 node list

# Información del nodo
ros2 node info /voice_assistant/agent_node

# Ver topics
ros2 topic list | grep voice_assistant
```

### Prueba Manual

Publica una transcripción simulada:

```bash
ros2 topic pub --once /voice_assistant/transcription_result std_msgs/msg/String \
  'data: "{\"text\": \"Hola, cuéntame sobre CAMI\", \"confidence\": 0.9, \"language\": \"es\"}"'
```

Escucha las respuestas:

```bash
ros2 topic echo /voice_assistant/llm_response
```

## Configuración

### Archivo de Configuración

Ubicación: `agent/config/agent_config.yaml`

```yaml
agent:
  ros__parameters:
    llm:
      # Provider: "openai" or "groq"
      provider: "groq"
      
      # Model name (provider-specific)
      # OpenAI: gpt-4o-mini, gpt-4o, gpt-3.5-turbo
      # Groq: llama-3.3-70b-versatile, mixtral-8x7b-32768
      model: "llama-3.3-70b-versatile"
      
      # LLM parameters
      temperature: 0.7
      max_tokens: 500
      streaming: true
      timeout: 10.0
      
    conversation:
      max_history_messages: 10
      
    # ... (resto de configuración)
```

### Instrucciones del Agente

Ubicación: `agent/prompts/cami_host_instructions.txt`

Este archivo contiene las instrucciones del sistema que definen:
- Personalidad del agente (carismático anfitrión)
- Información sobre CAMI (áreas, proyectos, misión)
- Directrices de comunicación
- Ejemplos de interacción

**Modifica este archivo** para ajustar el comportamiento del agente sin cambiar código.

## Estructura del Proyecto

```
agent/
├── agent/
│   ├── __init__.py
│   └── agent_node.py           # Nodo principal con LangChain
├── config/
│   └── agent_config.yaml       # Configuración ROS2
├── launch/
│   └── agent_node.launch.py    # Launch file
├── prompts/
│   └── cami_host_instructions.txt  # System prompt
├── package.xml                 # Metadatos del paquete
├── setup.py                    # Configuración de instalación
└── README.md                   # Esta documentación
```

## Desarrollo

### Modificar Proveedor y Modelo

Opciones:

1. **Editar config file**: Modifica `config/agent_config.yaml`
   ```yaml
   llm:
     provider: "groq"  # o "openai"
     model: "llama-3.3-70b-versatile"
   ```

2. **Override en launch**: Pasa parámetros al lanzar
   ```bash
   ros2 launch agent agent_node.launch.py provider:=openai model:=gpt-4o
   ```

### Debugging

#### Habilitar LangSmith Tracing

En `.env`:
```bash
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="lsv2_..."
LANGCHAIN_PROJECT="cami-voice-assistant"
```

Ver trazas en: https://smith.langchain.com/

#### Logs Detallados

```bash
ros2 launch agent agent_node.launch.py log_level:=debug
```

#### Ver Mensajes en Tiempo Real

```bash
# Terminal 1: Transcripciones
ros2 topic echo /voice_assistant/transcription_result

# Terminal 2: Respuestas LLM
ros2 topic echo /voice_assistant/llm_response
```

## Troubleshooting

### Error: "GROQ_API_KEY not found" o "OPENAI_API_KEY not found"

Asegúrate de que la variable del proveedor que usas esté configurada:

```bash
# Para Groq
echo $GROQ_API_KEY

# Para OpenAI
echo $OPENAI_API_KEY
```

Si no está disponible, añádela al `.env`:

```bash
echo 'export GROQ_API_KEY="gsk_..."' >> ~/.bashrc
source ~/.bashrc
```

### Error: "langchain_groq module not found"

Instala las dependencias:

```bash
pip install langchain-groq>=0.2.0
```

### Respuestas Lentas

1. **Usar Groq**: Groq es significativamente más rápido
   ```bash
   ros2 launch agent agent_node.launch.py provider:=groq
   ```

2. **Reducir max_tokens**: `max_tokens:=300`

3. **Usar modelo más rápido**:
   - Groq: `model:=llama-3.1-70b-versatile`
   - OpenAI: `model:=gpt-3.5-turbo`

### El Agente No Responde

1. Verifica que el nodo esté corriendo:
   ```bash
   ros2 node list | grep agent
   ```

2. Revisa los logs:
   ```bash
   ros2 launch agent agent_node.launch.py log_level:=debug
   ```

3. Verifica los topics:
   ```bash
   ros2 topic info /voice_assistant/transcription_result
   ros2 topic info /voice_assistant/llm_response
   ```

### Historial Conversacional No Funciona

El historial se mantiene durante la vida del nodo. Si reinicias el nodo, se pierde el contexto. Para persistencia entre sesiones, considera implementar:

- Checkpointing con LangGraph
- Base de datos externa para historial
- Redis/PostgreSQL para memoria a largo plazo

## Integración con el Pipeline Completo

### Orden de Lanzamiento

1. **voice_assistant_core**: Gestión de estado y audio
2. **stt**: Speech-to-Text
3. **agent**: LLM Agent (este paquete)
4. **tts**: Text-to-Speech (futuro)

### Launch File Global (Futuro)

```bash
ros2 launch voice_assistant_core full_pipeline.launch.py
```

## Mejoras Futuras

- [ ] Extracción automática de intents y entities
- [ ] Integración con base de conocimiento de CAMI
- [ ] Multi-agente para tareas especializadas
- [ ] Memoria a largo plazo (LangGraph stores)
- [ ] Soporte para herramientas/tools
- [ ] Análisis de sentimiento
- [ ] Respuestas multimodales (imágenes, diagramas)

## Referencias

- [LangChain Documentation](https://docs.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [ROS2 Documentation](https://docs.ros.org/en/jazzy/index.html)
- [Voice Assistant Core](../voice_assistant_core/README.md)

## Licencia

Apache-2.0

## Autor

**astra** - miguel.chumacero.b@gmail.com

---

*Desarrollado para el Centro Avanzado de Mecatrónica Inteligente (CAMI)*
