# Proyecto Candidates HRI 2026

El proyecto Candidates 2026 es un sistema de interacción humano-robot (HRI) construido sobre ROS 2 Jazzy dentro de un contenedor Docker en Ubuntu 24.04. El robot reconoce rostros, escucha preguntas por voz, consulta una base de conocimiento local y responde hablando en español.

## Seguridad y acceso

Solo las personas enroladas y autorizadas pueden recibir información del proyecto. Las personas desconocidas reciben un mensaje de acceso denegado.

## Arquitectura del sistema
El cerebro del robot funciona usando cinco nodos o sentidos conectados:
- **camera**: Sus ojos, capturando imágenes directamente desde la cámara web.
- **face_detector**: Usa YOLOv8 y ArcFace para detectar y reconocer exactamente quién le está hablando.
- **whisper_og**: Sus oídos, que usan Whisper y Silero VAD para detectar la actividad de voz y transcribirla al instante.
- **ollama**: Su motor de razonamiento, donde consulta una base de datos local usando ChromaDB y genera sus respuestas con el modelo Llama 3.2 de 1B.
- **voz**: Su boca, que usa Piper TTS para sintetizar la respuesta y hablar con una voz súper natural.

## Reglas de Seguridad
El robot solo comparte los detalles técnicos del proyecto con personas reconocidas visualmente y autorizadas. A las personas desconocidas les niega el acceso.

## El Creador: Alex
Alex es el único autor y desarrollador detrás del proyecto Candidates 2026. Como es un proyecto individual, Alex se encargó de construir absolutamente todo: desarrolló el pipeline de reconocimiento facial, armó el filtro anti-eco de voz, conectó el razonamiento de Llama 3.2, aisló toda la infraestructura en Docker y levantó el puente de telemetría con Foxglove para monitorear la visión sin lag. Su lenguaje favorito para unir toda esta magia es Python.

## Fecha Final de Entrega
Hoy, 9 de octubre de 2026, es la demostración final del proyecto Candidates HRI. El robot tiene que probar que funciona perfectamente reconociendo a Alex, manteniendo una conversación natural y contestando preguntas usando exclusivamente su memoria local.