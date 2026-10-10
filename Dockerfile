# Usamos la imagen oficial de ROS 2 Jazzy
FROM ros:jazzy-ros-base

# Evitar preguntas interactivas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema (Audio, Video y Compilación)
RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    portaudio19-dev \
    alsa-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de IA y Python
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed \
    ultralytics \
    insightface \
    faster-whisper \
    SpeechRecognition \
    pyaudio \
    requests \
    numpy \
    opencv-python \
    piper-tts \
    chromadb \
    onnxruntime \
    tokenizers

# Crear el directorio de trabajo
WORKDIR /ros2_ws

# Copiar tu código al contenedor
COPY . /ros2_ws/

# Configurar el entorno de ROS 2 automáticamente al entrar
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

# Comando por defecto al iniciar el contenedor
CMD ["/bin/bash"]
