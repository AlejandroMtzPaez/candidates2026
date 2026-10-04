import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from vision_interfaces.msg import FaceDetection
from vision_interfaces.srv import RagQuery

import threading
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel

class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_node')
        
        # evita eco
        self.ignore_echo = False
        self.create_subscription(Bool, '/robot_speaking', self.speaking_callback, 10)

        self.ultima_respuesta_robot = ""

        # face detection
        self.identidad_actual = "Desconocido"
        self.create_subscription(FaceDetection, '/vision/detections', self.face_callback, 10)
        
        # rag y voz
        self.rag_client = self.create_client(RagQuery, 'consultar_rag')
        self.tts_pub = self.create_publisher(String, '/robot_response', 10)

        # modelo whisper
        self.get_logger().info("Cargando modelo Whisper...")
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        
        # Speech Recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 5000 
        self.recognizer.dynamic_energy_threshold = False

        # HILO DEL MICRÓFONO
        self.thread = threading.Thread(target=self.listen_loop)
        self.thread.daemon = True
        self.thread.start()

    # ==========================================
    # CALLBACKS
    # ==========================================
    def face_callback(self, msg):
        self.identidad_actual = msg.name

    def speaking_callback(self, msg):
        self.ignore_echo = msg.data
        if self.ignore_echo:
            self.get_logger().info("Robot hablando. Descarta el audio.")

    def rag_response_callback(self, future):
        try:
            response = future.result()
            if response.access_granted:
                self.get_logger().info(f"Acceso validado: {response.answer}")
            else:
                self.get_logger().warn(f"Acceso denegado: {response.answer}")
            
            self.ultima_respuesta_robot = response.answer.strip().lower()

            msg_voz = String()
            msg_voz.data = response.answer
            self.tts_pub.publish(msg_voz)
        except Exception as e:
            self.get_logger().error(f"El servidor RAG falló: {e}")


    def listen_loop(self):
        with sr.Microphone() as source:
            self.get_logger().info("Ajustando al ruido ambiente...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.get_logger().info("Micrófono listo. Puedes hablar...")
            
            while rclpy.ok():
                if self.ignore_echo:
                    continue
                
                try:
                    # Escucha hasta que el volumen baje
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=15)
                    
                    # Si el robot empezó a hablar mientras escuchábamos, descartamos este audio
                    if self.ignore_echo:
                        continue
                        
                    self.get_logger().info("transcripción...")
                    
                    # Convertir audio crudo de sr a numpy para faster-whisper
                    raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    segments, _ = self.model.transcribe(audio_np, language="es")
                    texto = " ".join([seg.text for seg in segments]).strip()
                    
                    if texto and len(texto) > 5:
                        texto_limpio = texto.lower()

                        if self.ultima_respuesta_robot and (texto_limpio in self.ultima_respuesta_robot or self.ultima_respuesta_robot[:15] in texto_limpio):
                            self.get_logger().warn(f"Se elimina el eco {texto}")
                            continue


                        self.get_logger().info(f'[Tú dijiste]: "{texto}"')
                        self.get_logger().info(f"[Visión actual]: {self.identidad_actual}")
                        
                        #request para rag
                        req = RagQuery.Request()
                        req.identity = self.identidad_actual
                        req.question = texto
                        future = self.rag_client.call_async(req)
                        future.add_done_callback(self.rag_response_callback)
                        
                except sr.WaitTimeoutError:
                    # Pasa un segundo sin escuchar nada fuerte, reinicia el ciclo
                    pass
                except Exception as e:
                    self.get_logger().error(f"Error procesando audio: {e}")

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