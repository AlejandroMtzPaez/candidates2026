import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os
from std_msgs.msg import String, Bool

class TTSNode(Node):
    def __init__(self):
        super().__init__('voz')
        # se conecta a la respuesta final de ollama
        self.sub = self.create_subscription(String, '/robot_response', self.speak_callback, 10)
        # modelo español
        self.model_path = os.path.expanduser('~/ros2_ws/src/vision/vision/voices/es_MX-ald-medium.onnx')
        self.get_logger().info("La voz está lista.")

        # Publicador para avisar si el robot está hablando
        self.pub_is_speaking = self.create_publisher(Bool, '/is_speaking', 10)

    def speak_callback(self, msg):
        texto = msg.data
        self.get_logger().info(f"Voz: {texto}")

        # cuando el robot empieza a hablar
        speaking_msg = Bool()
        speaking_msg.data = True
        self.pub_is_speaking.publish(speaking_msg)


        wav_file = "/tmp/respuesta.wav"
        comando_piper = f"echo '{texto}' | piper --model {self.model_path} --output_file {wav_file}"

        try:
            # audio modelo ONNX
            subprocess.run(comando_piper, shell=True, check=True, stderr= subprocess.DEVNULL)

            # play audio por WSLg
            subprocess.run(f"aplay {wav_file}", shell=True, check=True, stderr=subprocess.DEVNULL)

        except Exception as e:
            self.get_logger().error(f"Error en la voz: {e}")


        finally:
            #cuando el robot ya no está hablando
            speaking_msg.data = False
            self.pub_is_speaking.publish(speaking_msg)

def main(args = None):
    rclpy.init(args = args)
    node = TTSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()