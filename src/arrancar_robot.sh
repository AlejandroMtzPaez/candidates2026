#!/usr/bin/env bash
set -e

WORKSPACE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "$WORKSPACE_DIR/install/setup.bash"

# Al hacer Ctrl+C se detienen todos los nodos hijos
trap 'kill 0' SIGINT SIGTERM

echo "Iniciando el sistema HRI Candidates..."
ros2 run vision camera &
ros2 run vision face_detector &
ros2 run vision whisper_og &
ros2 run vision ollama &
ros2 run vision voz &
wait
