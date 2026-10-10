# Modelos del Sistema

El proyecto Candidates utiliza un ensamble de modelos de inteligencia artificial especializados. Cada uno cumple una función específica dentro del pipeline de interacción:

## Visión por Computadora
- **YOLOv8-face:** Es el modelo encargado de la detección espacial. Su único trabajo es analizar los cuadros de la cámara y encontrar dónde hay rostros humanos en tiempo real.
- **ArcFace (InsightFace):** Es el modelo de reconocimiento biométrico. Toma los rostros encontrados por YOLO y extrae sus características faciales para identificar exactamente quién es la persona.

## Procesamiento de Audio
- **Silero VAD (Voice Activity Detection):** Es el filtro de atención del robot. Analiza el audio del micrófono para distinguir la voz humana del ruido de fondo, sabiendo exactamente cuándo alguien empieza y termina de hablar.
- **Faster-Whisper:** Es el modelo de transcripción (Speech-to-Text). Toma el audio filtrado por Silero y lo convierte en texto escrito con altísima precisión.

## Razonamiento y Voz
- **Llama 3.2 (1B):** Es el cerebro principal (LLM). Lee las preguntas transcritas, busca en su memoria (ChromaDB) y redacta respuestas coherentes y naturales. Al ser la versión de 1 billón de parámetros, es extremadamente rápido para correr en local.
- **Piper TTS:** Es el modelo de síntesis de voz (Text-to-Speech). Recibe el texto generado por Llama y lo transforma en un archivo de audio hablado con una voz fluida y humana, completando el ciclo de interacción.