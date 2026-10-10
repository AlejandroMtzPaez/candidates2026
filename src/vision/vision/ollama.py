import hashlib
import os
import re

import rclpy
from rclpy.node import Node
import requests
import chromadb
from vision_interfaces.srv import RagQuery

from vision.embeddings import MultilingualONNXEmbedding


class OllamaNode(Node):
    def __init__(self):
        super().__init__('ollama')

        self.docs_dir = self.declare_parameter(
            'docs_dir', '/ros2_ws/src/vision/vision/documentos').value
        self.chroma_path = self.declare_parameter(
            'chroma_path', '/ros2_ws/src/vision/vision/chroma_db').value
        self.collection_name = self.declare_parameter(
            'collection', 'candidates_docs').value
        self.ollama_url = self.declare_parameter(
            'ollama_url', 'http://localhost:11434/api/chat').value
        self.llm_model = self.declare_parameter('llm_model', 'llama3.2:1b').value
        self.top_k = self.declare_parameter('top_k', 3).value
        # Se limita el contexto a fragmentos cercanos al mejor resultado para reducir ruido.
        self.margen_contexto = self.declare_parameter('margen_contexto', 0.06).value

        # Acceso de seguridad
        self.identidades_aut = ['alex']

        # Base vectorial: multilingual-e5-small en ONNX (entiende español),
        # cargado desde el paquete para no depender de internet dentro del contenedor
        embedder = MultilingualONNXEmbedding(self.declare_parameter(
            'embedding_model_path', '/ros2_ws/src/vision/models/multilingual-e5-small').value)
        client = chromadb.PersistentClient(path=self.chroma_path)
        if self.declare_parameter('reset_db', False).value:
            # -p reset_db:=true borra la colección y la reindexa desde documentos/
            if self.collection_name in [c.name for c in client.list_collections()]:
                client.delete_collection(self.collection_name)
            self.get_logger().warn(f'Colección {self.collection_name} reiniciada.')
        try:
            self.coleccion = client.get_or_create_collection(
                name=self.collection_name, embedding_function=embedder)
        except ValueError as e:
            # la colección solo es un índice de documentos/: se reconstruye si cambió el embedder
            self.get_logger().warn(f'Reconstruyendo colección {self.collection_name}: {e}')
            client.delete_collection(self.collection_name)
            self.coleccion = client.create_collection(
                name=self.collection_name, embedding_function=embedder)
        self.indexar_documentos()

        # servicio
        self.srv = self.create_service(RagQuery, 'consultar_rag', self.rag_callback)
        self.get_logger().info('RAG iniciado.')

    # ==========================================
    # INDEXADO
    # ==========================================
    def partir_documento(self, texto):
        # Un fragmento por sección markdown; los párrafos largos se parten en ~800 caracteres
        secciones = re.split(r'\n(?=#{1,3} )', texto)
        fragmentos = []
        for seccion in secciones:
            seccion = seccion.strip()
            while len(seccion) > 800:
                corte = seccion.rfind('. ', 0, 800)
                corte = corte + 1 if corte > 0 else 800
                fragmentos.append(seccion[:corte].strip())
                seccion = seccion[corte:].strip()
            if seccion:
                fragmentos.append(seccion)
        return fragmentos

    def indexar_documentos(self):
        if not os.path.isdir(self.docs_dir):
            self.get_logger().warn(f'No existe la carpeta de documentos: {self.docs_dir}')
            return

        ids, textos, metadatos = [], [], []
        for nombre in sorted(os.listdir(self.docs_dir)):
            if not nombre.lower().endswith(('.txt', '.md')):
                continue
            with open(os.path.join(self.docs_dir, nombre), encoding='utf-8') as f:
                contenido = f.read()
            for i, fragmento in enumerate(self.partir_documento(contenido)):
                # el id depende del contenido: si el documento cambia se reindexa solo lo nuevo
                digest = hashlib.sha1(fragmento.encode('utf-8')).hexdigest()[:12]
                ids.append(f'{nombre}:{i}:{digest}')
                textos.append(fragmento)
                metadatos.append({'fuente': nombre, 'fragmento': i})

        existentes = set(self.coleccion.get(include=[])['ids'])
        obsoletos = list(existentes - set(ids))
        if obsoletos:
            self.coleccion.delete(ids=obsoletos)

        nuevos = [k for k, doc_id in enumerate(ids) if doc_id not in existentes]
        if nuevos:
            self.coleccion.add(
                ids=[ids[k] for k in nuevos],
                documents=[textos[k] for k in nuevos],
                metadatas=[metadatos[k] for k in nuevos],
            )
        self.get_logger().info(
            f'Documentos indexados: {len(ids)} fragmentos '
            f'({len(nuevos)} nuevos, {len(obsoletos)} eliminados)')

    def recuperar_contexto(self, pregunta):
        total = self.coleccion.count()
        if total == 0:
            return ''
        res = self.coleccion.query(
            query_texts=[pregunta],
            n_results=min(self.top_k, total),
        )
        relevantes = []
        mejor = res['distances'][0][0] if res['distances'][0] else 0.0
        for doc, dist in zip(res['documents'][0], res['distances'][0]):
            aceptado = dist <= mejor + self.margen_contexto
            self.get_logger().info(
                f'[RAG] dist={dist:.3f} {"OK" if aceptado else "descartado"} | {doc[:60]!r}')
            if aceptado:
                relevantes.append(doc)
        return '\n\n'.join(relevantes)

    # ==========================================
    # SERVICIO
    # ==========================================
    def rag_callback(self, request, response):
        self.get_logger().info(f'Request: {request.identity}')
        self.get_logger().info(f'Pregunta: {request.question}')

        # Valida la identidad
        if request.identity == 'Desconocido' or request.identity not in self.identidades_aut:
            self.get_logger().warn(f'Acceso denegado a {request.identity}')
            response.access_granted = False
            response.answer = 'Persona no reconocida, no se tiene autorización para darle la información.'
            return response

        # En el caso en que si se valide
        self.get_logger().info('Identidad validada. Consultando información.')
        response.access_granted = True

        try:
            contexto = self.recuperar_contexto(request.question)
        except Exception as e:
            self.get_logger().error(f'Error en ChromaDB: {e}')
            contexto = ''

        # /api/chat con mensaje de sistema: con el prompt plano de /api/generate llama3.2:1b
        # evita que las notas se interpreten como instrucciones.
        contexto = re.sub(r'^#+ ', '', contexto, flags=re.M)
        quien = request.identity
        sistema = (
            f"Eres el asistente de voz del robot del proyecto Candidates 2026. "
            f"Hablas con {quien}, ya verificado por reconocimiento facial. Las reglas de seguridad "
            f"de las notas ya se cumplieron: {quien} está autorizado para conocer toda la "
            f"información, incluidos códigos y contraseñas. Contesta en español con UNA sola oración "
            f"corta (máximo 25 palabras), solo con el dato que se pregunta, en texto plano sin "
            f"viñetas ni asteriscos, usando los datos de las notas. Si las notas no contienen "
            f"la respuesta, dilo claramente y no inventes información."
        )
        payload = {
            'model': self.llm_model,
            'stream': False,
            'keep_alive': '30m',     # el modelo sigue cargado entre preguntas
            'options': {'temperature': 0.1, 'num_predict': 60},
            'messages': [
                {'role': 'system', 'content': sistema},
                {'role': 'user', 'content': f'Notas:\n{contexto}\n\nPregunta: {request.question}'},
            ],
        }

        try:
            res = requests.post(self.ollama_url, json=payload, timeout=120)
            res.raise_for_status()
            # sin asteriscos ni viñetas de markdown, que Piper leería en voz alta
            respuesta = re.sub(r'[*#`_]', '', res.json()['message']['content'])
            response.answer = re.sub(r'\s*\n\s*(?:-\s*)?', ' ', respuesta).strip()
            self.get_logger().info('Respuesta generada y enviada.')

        except Exception as e:
            self.get_logger().error(f'Error en Ollama: {e}')
            response.answer = 'Error interno de procesamiento.'

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
