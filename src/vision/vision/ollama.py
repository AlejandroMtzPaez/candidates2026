import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import json
import os
import chromadb
from vision_interfaces.srv import RagQuery

class OllamaNode(Node):
    def __init__(self):
        super().__init__('ollama')
        
        # servicio
        self.srv = self.create_service(RagQuery, 'consultar_rag', self.rag_callback)

        # Acceso de seguridad
        self.identidades_aut = ["alex"]
        self.get_logger().info("RAG iniciado.")

    def rag_callback(self, request, response):
        self.get_logger().info(f"Request: {request.identity}")
        self.get_logger().info(f"Pregunta: {request.question}")

        # Valida la identidad
        if request.identity == "Desconocido" or request.identity not in self.identidades_aut:
            self.get_logger().warn(f"Acceso denegado a {request.identity}")
            response.access_granted = False
            response.answer = "Persona no reconocida, no se tiene autorización para darle la información."
            return response
        
        # En el caso en que si se valide
        self.get_logger().info("Identidad validada. Consultando información.")
        response.access_granted = True

        # RAG o contexto
        contexto = "El proyecto Candidates está seguro"

        prompt = (
            f"Eres el asistente de voz oficial de este proyecto. "
            f"ESTÁS AUTORIZADO para revelar cualquier información del contexto proporcionado. "
            f"NO te disculpes. NO digas que no puedes proporcionar información personal. "
            f"Responde de manera directa, breve y usando SOLAMENTE este contexto.\n\n"
            f"Contexto: {contexto}\n"
            f"Pregunta de {request.identity}: {request.question}\n"
            f"Respuesta directa:"
        )

        payload = {
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False
        }

        try:
            res = requests.post("http://localhost:11434/api/generate", json=payload)
            res.raise_for_status()
            response.answer = res.json()['response'].strip()
            self.get_logger().info("Respuesta generada y enviada.")

        except Exception as e:
            self.get_logger().error(f"Error en Ollama: {e}")
            response.answer = "Error interno de procesamiento."

        return response


def main(args=None):
    rclpy.init(args=args)
    node = OllamaNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()