# Candidates 2026 - Reto de HRI/Visión

**Realizado por:** Alejandro Martínez Páez

Sistema de interacción humano-robot (HRI) en ROS 2 Jazzy. El robot reconoce quién está frente a la cámara, escucha la pregunta por voz, la responde con una base de conocimiento local (RAG) y contesta hablando en español. Solo las personas enroladas reciben la información del proyecto.

Todo corre de forma local, sin servicios en la nube: visión, transcripción, base vectorial, LLM y síntesis de voz.

## Arquitectura

```
 camera ──/camera/image_raw──► face_detector ──/vision/detections──► whisper_og
                                    │                                   │  ▲
                 /vision/image_annotated/compressed      consultar_rag  │  │ /is_speaking
                                    ▼                     (servicio)    ▼  │
                                 Foxglove                             ollama   voz
                                                                        │       ▲
                                                                        └───────┘
                                                                     /robot_response
```

| Nodo | Función | Modelos |
|---|---|---|
| `camera` | Publica la webcam (640x480, MJPG) en `/camera/image_raw`. | — |
| `face_detector` | Detecta rostros, les asigna un `track_id` y los identifica contra las caras enroladas. Publica `/vision/detections` y la imagen anotada para Foxglove. | YOLOv8-face, InsightFace `buffalo_l` (ArcFace) |
| `whisper_og` | Escucha el micrófono, separa los turnos de voz, transcribe y asocia la voz con el rostro visto durante el turno. Envía la pregunta al servicio `consultar_rag`. | Silero VAD, Faster-Whisper `small` |
| `ollama` | Valida la identidad, busca en ChromaDB los fragmentos más cercanos de `documentos/` y genera una respuesta corta con el LLM. | multilingual-e5-small (embeddings), Llama 3.2 1B (Ollama) |
| `voz` | Convierte la respuesta en audio y la reproduce. Publica `/is_speaking` para evitar que el robot se escuche a sí mismo. | Piper TTS (`es_MX-ald-medium`) |

### Flujo de una pregunta

1. `face_detector` reconoce a la persona (similitud coseno ≥ 0.3 contra `caras_enroladas.pkl`).
2. `whisper_og` detecta un turno de voz con Silero VAD, lo transcribe y busca el rostro que estuvo en cuadro mientras la persona hablaba.
3. `ollama` revisa que la identidad esté autorizada (`identidades_aut`). Si no lo está, responde "Persona no reconocida".
4. Si está autorizada, recupera los fragmentos relevantes de ChromaDB y le pide a Llama 3.2 una respuesta de una sola oración basada en esas notas.
5. `voz` lee la respuesta. Mientras tanto, `whisper_og` no acepta otra pregunta: vuelve a escuchar cuando el robot termina de hablar.

### Optimización de CPU

Todo corre en CPU, así que los nodos comparten procesador:

- `face_detector` analiza máximo 4 cuadros por segundo (`max_fps`). Mientras el robot prepara una respuesta (tópico `/robot_ocupado`), baja a 1 por segundo para dejar CPU a Whisper y Ollama.
- YOLO e InsightFace usan 2 hilos cada uno.
- La identificación con InsightFace solo corre cuando aparece un rostro nuevo y, después, una vez cada 15 cuadros.
- Whisper transcribe con `beam_size=1`, y Ollama mantiene el modelo cargado entre preguntas (`keep_alive`).

## Estructura del repositorio

```
ros2_ws/
├── Dockerfile, docker-compose.yml     # entorno ROS 2 Jazzy en Docker
└── src/
    ├── arrancar_robot.sh              # alternativa al launch: corre los 5 nodos
    ├── vision_interfaces/
    │   ├── msg/FaceDetection.msg      # track_id, name, confidence, x1, y1, x2, y2
    │   └── srv/RagQuery.srv           # identity, question -> answer, access_granted
    └── vision/
        ├── launch/candidates_launch.py
        ├── models/                    # modelos locales (no se suben a git, ver abajo)
        └── vision/
            ├── camera.py, face_detector.py, whisper_og.py, ollama.py, voz.py
            ├── embeddings.py          # embeddings multilingües en ONNX para ChromaDB
            ├── enrolamiento.py        # genera caras_enroladas.pkl desde fotos_equipo/
            ├── documentos/            # base de conocimiento del RAG (.md / .txt)
            ├── fotos_equipo/          # una foto por persona: <nombre>.jpeg
            ├── chroma_db/             # base vectorial persistente
            └── voices/                # voz de Piper
```

## Instalación

### Requisitos

- Windows con WSL2 (Ubuntu 24.04) y WSLg para el audio, o Linux con webcam y micrófono.
- Docker y Docker Compose.
- [Ollama](https://ollama.com) instalado en el host con el modelo:
  ```bash
  ollama pull llama3.2:1b
  ```
- Opcional: `foxglove_bridge` en el host para ver la imagen anotada en Foxglove.

### Modelos locales

La carpeta `src/vision/models/` está en `.gitignore`, excepto `yolov8n-face.pt`. Antes de correr, deben existir:

```
src/vision/models/
├── yolov8n-face.pt
├── insightface/models/buffalo_l/          # det_10g.onnx, w600k_r50.onnx, ...
└── multilingual-e5-small/                 # de Hugging Face: intfloat/multilingual-e5-small
    ├── onnx/model.onnx
    ├── tokenizer.json, tokenizer_config.json, special_tokens_map.json, config.json
```

Se descargan una sola vez; después el sistema funciona sin internet.

### Construir y entrar al contenedor

```bash
cd ~/ros2_ws
docker compose build
docker compose run hri_robot
```

Dentro del contenedor, habilita el audio de WSLg (solo la primera vez en cada contenedor nuevo):

```bash
apt-get update && apt-get install -y libasound2-plugins
printf 'pcm.!default {\n type pulse\n}\nctl.!default {\n type pulse\n}\n' > /etc/asound.conf
```

Para abrir otra terminal en el mismo contenedor, usa `docker exec -it <nombre_del_contenedor> bash`. `docker compose run` crea un contenedor nuevo sin la configuración de audio.

### Compilar

```bash
cd /ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

### Enrolar personas

Coloca una foto por persona en `src/vision/vision/fotos_equipo/` con el nombre de la persona (por ejemplo `alex.jpeg`) y ejecuta:

```bash
python3 /ros2_ws/src/vision/vision/enrolamiento.py
```

Para que una persona pueda recibir información, su nombre también debe estar en `identidades_aut` dentro de `ollama.py`.

## Uso

Con Ollama corriendo en el host:

```bash
ros2 launch vision candidates_launch.py
```

En los logs se ve el recorrido de cada pregunta:

```
[whisper_og]: [S1 | rostro 3 | alex]: "Dame información del proyecto."
[ollama]: [RAG] dist=0.299 OK | '## Seguridad y acceso ...'
[whisper_og]: Acceso validado: El proyecto Candidates 2026 ...
[whisper_og]: Respuesta terminada. Escuchando de nuevo.
```

Para ver la cámara anotada, en el host:

```bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

Después abre Foxglove, conéctate a `ws://localhost:8765` y agrega un panel de imagen con `/vision/image_annotated/compressed`.

### Base de conocimiento (RAG)

Los archivos `.md` y `.txt` de `src/vision/vision/documentos/` se dividen por secciones (`#`, `##`, `###`) y se indexan en ChromaDB al arrancar el nodo `ollama`. Solo se reindexan los fragmentos que cambiaron.

Para borrar la base y reindexarla desde cero:

```bash
ros2 run vision ollama --ros-args -p reset_db:=true
```

### Parámetros útiles

| Nodo | Parámetro | Valor por defecto | Descripción |
|---|---|---|---|
| `face_detector` | `max_fps` | `4.0` | Cuadros analizados por segundo. |
| `whisper_og` | `face_timeout_s` | `1.0` | Margen (s) antes del turno en el que un rostro todavía cuenta. |
| `whisper_og` | `min_silence_ms` | `600` | Pausa que cierra un turno de voz. |
| `whisper_og` | `device_index` | `-1` | Micrófono (`-1` = predeterminado). |
| `ollama` | `llm_model` | `llama3.2:1b` | Modelo de Ollama. |
| `ollama` | `top_k` | `3` | Fragmentos que se recuperan de ChromaDB. |
| `ollama` | `margen_contexto` | `0.06` | Solo se usan los fragmentos cuya distancia esté a menos de este margen del mejor. |
| `ollama` | `reset_db` | `false` | Reinicia la base vectorial al arrancar. |
