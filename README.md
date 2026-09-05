# Candidates 2026 — Reto de HRI/Visión
Implementación del Proyecto

**Realizado por:** Alejandro Martínez Páez

## Descripción (Objetivo final)

Sistema en ROS 2 que detecta e identifica personas frente a una cámara, y responderá preguntas usando un documento privado por persona (RAG), fusionando identidad facial, voz y movimiento de boca.

## Instalación
Este proyecto usa ROS 2 Jazzy sobre Ubuntu 24.04 (probado en WSL2).

1. Instalar ROS 2 Jazzy siguiendo la [guía oficial](https://docs.ros.org/en/jazzy/Installation.html).

2. Instalar dependencias adicionales:
```bash
   sudo apt install ros-jazzy-cv-bridge ros-jazzy-vision-opencv python3-opencv ros-jazzy-rqt-image-view -y
```
3. Clonar este repositorio dentro de un workspace de ROS 2:
```bash
   git clone <URL_DE_ESTE_REPO> ~/ros2_ws
   cd ~/ros2_ws
   colcon build
   source install/setup.bash
```

## Estructura del repositorio

- `src/vision/` — paquete principal con los nodos de percepción
  - `hello_node.py` — nodo mínimo de práctica para el entendimiento de las bases a utilizar.
  - `camera.py` — nodo publisher de cámara (approx. 5 fps).
- `src/vision_interfaces/` — interfaces propias del proyecto:
  - `msg/FaceDetection.msg` — mensaje con datos de detección (track_id, name, confidence, coordinates).

## Probar el avance

```bash
ros2 run vision camera
```

En otra terminal, para verificar que publica:
```bash
ros2 run vision camera_test_subscriber
```