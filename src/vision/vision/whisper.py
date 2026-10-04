import rclpy
from rclpy.node import Node
from std_msgs.msg import String
#import speech_recognition as sr
from faster_whisper import WhisperModel
import numpy as np
import threading
from std_msgs.msg import String, Bool
from vision_interfaces.msg import FaceDetection
from vision_interfaces.srv import RagQuery
import torch
import pyaudio


class WhisperNode(Node):
    def __init__(self):
        super().__init__("whisper")

        # evitar ECO
        self.ignore_echo = False
        self.create_subscription(Bool, '/robot_speaking', self.speaking_callback, 10)
        
        # Visión
        self.identidad_actual = "Desconocido"
        self.create_subscription(FaceDetection, '/vision/detections', self.face_callback, 10)

        # RAG
        self.rag_client = self.create_client(RagQuery, 'consultar_rag')

        # Voz
        self.tts_pub = self.create_publisher(String, '/robot_response', 10)
        
        # Audio
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        
        self.get_logger().info("Cargando Silero VAD...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            force_reload=False
        )
        self.vad_model.eval()

        # Configuracion de PyAudio
        self.RATE = 16000
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1

        # Se ejecuta en un hilo separado para no bloquear los callbacks de ROS 2
        self.thread = threading.Thread(target=self.listen_loop)
        self.thread.daemon = True
        self.thread.start()

    def face_callback(self, msg):
        # se guarda la última cara
        self.identidad_actual = msg.name

    def speaking_callback(self, msg):
        # Si el mensaje es True, el robot empezó a hablar. Todo lo que grabe el micro se ignora
        self.ignore_echo = msg.data
        if self.ignore_echo:
            self.get_logger().info("Robot hablando: ignora el micrófono.")

    def listen_loop(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=self.CHANNELS, 
                        rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        
        self.get_logger().info("Audio listo. Esperando voz...")
        
        audio_buffer = []
        is_speaking = False
        silence_chunks = 0
        # 1 segundo de silencio para terminar de hablar
        MAX_SILENCE = int(self.RATE / self.CHUNK * 1) 
        
        while rclpy.ok():
            try:
                # lee el audio
                data = stream.read(self.CHUNK, exception_on_overflow=False)

                if self.ignore_echo: # para evitar el eco con la voz del bot
                    is_speaking = False
                    audio_buffer = []
                    silence_chunks = 0
                    continue

                audio_int16 = np.frombuffer(data, dtype=np.int16)
                
                # floats entre -1.0 y 1.0 para silero
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                tensor_chunk = torch.from_numpy(audio_float32)
                
                # el modelo silero evalúa el audio
                prob = self.vad_model(tensor_chunk, self.RATE).item()
                
                # Si es voz humana con más del 50% de probabilidad
                if prob > 0.5:
                    if not is_speaking:
                        self.get_logger().info("🗣️ Voz detectada. Grabando...")
                        is_speaking = True
                    audio_buffer.append(audio_float32)
                    silence_chunks = 0 # restart el contador de silencio
                
                # Si está grabando pero este pedacito fue silencio
                elif is_speaking:
                    audio_buffer.append(audio_float32)
                    silence_chunks += 1
                    
                    # Corte de grabación si se detecta silencio
                    if silence_chunks > MAX_SILENCE:
                        self.get_logger().info("Transcripción...")
                        is_speaking = False
                        
                        # junta los pedazos grabados
                        final_audio = np.concatenate(audio_buffer)
                        audio_buffer = []
                        silence_chunks = 0
                        
                        # WHISPER TRANSCRIBE
                        segments, _ = self.model.transcribe(final_audio, language="es")
                        texto = " ".join([seg.text for seg in segments]).strip()
                        
                        if texto and len(texto) > 5:
                            self.get_logger().info(f'[Tú dijiste]: "{texto}"')
                            self.get_logger().info(f"[Visión actual]: {self.identidad_actual}")
                            
                            # request para el RAG
                            req = RagQuery.Request()
                            req.identity = self.identidad_actual
                            req.question = texto
                            future = self.rag_client.call_async(req)
                            future.add_done_callback(self.rag_response_callback)
            
            except Exception as e:
                self.get_logger().error(f"Error en el ciclo de audio: {e}")

    def rag_response_callback(self, future):
        try:
            response = future.result()
            if response.access_granted:
                self.get_logger().info("Acceso validado.")
            else:
                self.get_logger().warn("Acceso denegado")

            # mensaje a piper para decirlo en voz
            msg_voz = String()
            msg_voz.data = response.answer
            self.tts_pub.publish(msg_voz)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = WhisperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
