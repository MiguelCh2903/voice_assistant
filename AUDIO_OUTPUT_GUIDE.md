# 🔊 Manejo de Audio con VAPI - Explicación y Soluciones

## 📋 Estado Actual

### ✅ Lo que funciona:
- Conexión a ESPHome (micrófono)
- Captura de audio del micrófono ESPHome
- Envío potencial a VAPI (cuando la key esté correcta)

### ❌ Problema Principal:
**El audio de respuesta de VAPI sale en los speakers de tu PC, NO en ESPHome**

---

## 🎯 Arquitectura Actual

```
┌─────────────────┐
│  ESPHome Device │
│  (Micrófono)    │
└────────┬────────┘
         │ Audio IN (aioesphomeapi)
         ▼
┌─────────────────────────┐
│ vapi_voice_assistant    │
│       Node              │
│  ┌──────────────────┐   │
│  │  ESPHomeClient   │   │
│  └─────────┬────────┘   │
│            │             │
│  ┌─────────▼────────┐   │
│  │   VapiClient     │   │
│  │   (Vapi SDK)     │   │
│  └─────────┬────────┘   │
└────────────┼────────────┘
             │ WebRTC/Daily.co
             ▼
┌─────────────────────────┐
│     VAPI Cloud API      │
│  (STT → LLM → TTS)      │
└─────────────────────────┘
             │
             ▼ Audio OUT via Daily.co
┌─────────────────────────┐
│   PC Speakers 🔊        │ ← AQUÍ SALE EL AUDIO
│   (NO ESPHome)          │
└─────────────────────────┘
```

---

## 🔧 Soluciones Posibles

### Opción 1: 🎧 Usar VAPI Solo para STT + LLM (Recomendado para ESPHome)

**Concepto**: Usar VAPI solo para transcripción y respuesta de texto, luego usar TTS local o del mismo ESPHome.

**Pros**:
- Audio sale directamente en ESPHome
- Control total del flujo de audio
- Menor latencia en el dispositivo

**Contras**:
- Necesitas implementar TTS separado
- No aprovechas el TTS de VAPI

### Opción 2: 🔀 Capturar Audio de VAPI y Reenviarlo a ESPHome

**Concepto**: Interceptar el audio de salida de Daily.co y reenviarlo a ESPHome.

**Pros**:
- Usas el TTS de VAPI
- Audio sale en ESPHome

**Contras**:
- Complejo de implementar
- Requiere acceso a internals de Daily.co SDK
- Posible latencia adicional

### Opción 3: 🌐 Usar VAPI Web SDK en lugar de Python SDK

**Concepto**: En lugar del Python SDK, usar la API REST de VAPI directamente.

**Pros**:
- Control total del audio
- Puedes procesar el audio de respuesta como quieras

**Contras**:
- Más trabajo de implementación
- Necesitas manejar WebSocket manualmente

### Opción 4: 📱 Configuración Híbrida (Simple)

**Concepto**: Usa tu PC/ROS2 como "control center" y el audio sale por allí temporalmente.

**Pros**:
- Funciona inmediatamente
- Útil para desarrollo/testing

**Contras**:
- El audio no sale en ESPHome

---

## 🚀 Recomendación: Implementación Práctica

Para tu caso de uso con ESPHome, te recomiendo **Opción 2 mejorada**:

### Solución: Streaming bidireccional

Modificar el `VapiClient` para:

1. **Audio IN (ESPHome → VAPI)**: ✅ Ya funciona
2. **Audio OUT (VAPI → ESPHome)**: ⚠️ Necesita implementarse

#### Cambios necesarios:

**1. Acceder al Daily.co client interno de VAPI**

El SDK de VAPI usa `DailyCall` internamente. Necesitamos:
- Acceder al stream de audio del speaker
- Capturar los frames de audio
- Enviarlos a ESPHome

**2. ESPHome debe poder recibir audio para TTS**

Verifica que tu dispositivo ESPHome soporte:
```yaml
voice_assistant:
  microphone: mic_id
  speaker: speaker_id  # ← Necesitas esto
```

**3. Implementar el loop de audio OUT**

Similar al loop de audio IN, pero al revés.

---

## 💡 Quick Fix para Empezar

Mientras decides qué implementar, puedes:

### 1. Verificar la Public Key

```bash
# Edita ~/.env
export VAPI_API_KEY="pk_xxxxxxxx"  # Debe empezar con "pk_" (public key)
# NO uses "sk_xxxxxxxx" (secret key)

source ~/.env
```

### 2. Probar con Audio en PC (temporalmente)

Para verificar que todo funciona, puedes:

```bash
# El audio saldrá en tu PC
ros2 launch voice_assistant_core vapi_voice_assistant.launch.py
```

Habla al micrófono ESPHome, y escucha la respuesta en los speakers de tu PC.

### 3. Verificar ESPHome Speaker

```bash
# Conéctate al dispositivo ESPHome
esphome logs home-assistant-voice-0a5339.yaml

# Verifica que tenga speaker configurado
# Debería aparecer algo como:
# [speaker:XXX] Speaker configured
```

---

## 📝 Siguiente Paso: Implementar Audio OUT

¿Quieres que te ayude a implementar el streaming de audio de VAPI a ESPHome?

Necesitaré:
1. ✅ Confirmar que tu ESPHome tiene speaker configurado
2. ✅ Revisar la API de aioesphomeapi para enviar audio
3. ✅ Modificar `VapiClient` para capturar audio OUT
4. ✅ Crear el loop de streaming a ESPHome

---

## 🔍 Debugging: Ver el Audio Flow

Para entender dónde está el audio:

```python
# Añade esto temporalmente en vapi_client.py
def _start_call_blocking(...):
    self._vapi.start(...)
    
    # Acceder al Daily client interno
    if hasattr(self._vapi, '_client'):
        daily_client = self._vapi._client
        print(f"Daily client: {daily_client}")
        print(f"Daily speaker device: {getattr(daily_client, '_DailyCall__speaker_device', None)}")
```

Esto te dirá si tienes acceso al dispositivo de audio de Daily.co.

---

**¿Qué prefieres hacer?**

1. **Quick test**: Solo arreglar la API key y probar con audio en PC
2. **Implementación completa**: Streaming bidireccional ESPHome ↔ VAPI
3. **Alternativa**: Cambiar a usar solo VAPI para STT/LLM, y TTS local

Dime qué opción prefieres y te ayudo a implementarla. 🚀
