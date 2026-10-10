import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String, Bool
from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import os
import onnxruntime as ort
import torch
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from vision_interfaces.msg import FaceDetection

# sin límite YOLO e InsightFace usan todos los núcleos y Ollama se queda sin CPU (timeout)
torch.set_num_threads(2)
OPCIONES_ORT = ort.SessionOptions()
OPCIONES_ORT.intra_op_num_threads = 2


def similitud_coseno(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


class FaceDetector(Node):
    def __init__(self):
        super().__init__('face_detector')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, qos
        )
        self.detection_pub = self.create_publisher(FaceDetection, 'vision/detections', 10)
        # depth=1: Foxglove siempre recibe el frame más reciente, sin acumular cola
        self.annotated_pub = self.create_publisher(
            CompressedImage, '/vision/image_annotated/compressed', 1)
        self.jpeg_quality = self.declare_parameter('jpeg_quality', 60).value

        # última frase de cada rostro (whisper_og publica el hablante asociado al track)
        self.ultimo_habla = {}      # track_id -> (texto, speaker, t)
        self.create_subscription(String, '/transcription', self.transcription_callback, 10)

        # no hace falta analizar cada frame: máximo 4 por segundo, y 1 por segundo mientras
        # el robot procesa una respuesta (así Whisper y Ollama tienen CPU)
        self.periodo_normal = 1.0 / self.declare_parameter('max_fps', 4.0).value
        self.periodo_ocupado = 1.0
        self.robot_ocupado = False
        self.ultimo_frame = 0.0
        self.create_subscription(Bool, '/robot_ocupado', self.ocupado_callback, 10)

        model_path = self.declare_parameter(
            'model_path',
            '/ros2_ws/src/vision/models/yolov8n-face.pt'
        ).value
        self.model = YOLO(model_path)
        self.get_logger().info('Modelo YOLOv8-face cargado correctamente')

        # modelos de InsightFace dentro del paquete (sin descargas en el contenedor);
        # solo detección + reconocimiento: landmarks y edad/género no se usan
        insight_root = self.declare_parameter(
            'insightface_root', '/ros2_ws/src/vision/models/insightface').value
        self.face_app = FaceAnalysis(name='buffalo_l', root=insight_root,
                                     allowed_modules=['detection', 'recognition'],
                                     sess_options=OPCIONES_ORT)
        self.face_app.prepare(ctx_id=0, det_size=(320, 320))

        self.db_embeddings = {}
        db_path = os.path.expanduser('/ros2_ws/src/vision/vision/caras_enroladas.pkl')
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                self.db_embeddings = pickle.load(f)
            self.get_logger().info(f"Identidades cargadas: {list(self.db_embeddings.keys())}")
        self.umbral_identidad = 0.3

        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = 0.3

        self.frame_count = 0
        self.identity_cache = {}

    def transcription_callback(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if d.get('track_id', -1) >= 0:
            self.ultimo_habla[d['track_id']] = (d.get('text', ''), d.get('speaker', ''), time.time())

    def ocupado_callback(self, msg):
        self.robot_ocupado = msg.data

    def image_callback(self, msg):
        ahora = time.time()
        periodo = self.periodo_ocupado if self.robot_ocupado else self.periodo_normal
        if ahora - self.ultimo_frame < periodo:
            return
        self.ultimo_frame = ahora

        # copia escribible del buffer (respeta msg.step por si la fila trae padding)
        buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        frame = buf[:, :msg.width * 3].reshape(msg.height, msg.width, 3).copy()

        results = self.model.predict(frame, verbose=False, conf=0.4, imgsz=320)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
        current_frame_ids = self.assign_track_ids(boxes)

        self.frame_count += 1
        # InsightFace es lo más pesado: solo con rostros en cuadro, de inmediato si aparece
        # un track nuevo y, para los ya reconocidos, una verificación cada 15 frames
        run_identity_this_frame = len(boxes) > 0 and (
            (self.frame_count % 15 == 0) or any(t not in self.identity_cache for t in current_frame_ids))

        insight_faces = []
        if run_identity_this_frame:
            insight_faces = self.face_app.get(frame)

        for box, track_id in zip(boxes, current_frame_ids):
            x1, y1, x2, y2 = box

            if run_identity_this_frame:
                nombre_detectado = 'Desconocido'
                confianza_identidad = 0.0
                cx_yolo, cy_yolo = (x1 + x2) / 2, (y1 + y2) / 2
                mejor_iface, menor_distancia = None, float('inf')

                for iface in insight_faces:
                    ix1, iy1, ix2, iy2 = iface.bbox
                    cx_iface, cy_iface = (ix1 + ix2) / 2, (iy1 + iy2) / 2
                    distancia = ((cx_yolo - cx_iface) ** 2 + (cy_yolo - cy_iface) ** 2) ** 0.5
                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        mejor_iface = iface

                if mejor_iface is not None and menor_distancia < 150:
                    embedding_actual = mejor_iface.embedding
                    mejor_score = 0.0
                    for nombre_db, emb_db in self.db_embeddings.items():
                        score = similitud_coseno(embedding_actual, emb_db)
                        if score > mejor_score:
                            mejor_score = score
                            nombre_detectado = nombre_db if score >= self.umbral_identidad else 'Desconocido'
                    confianza_identidad = float(mejor_score)

                # un frame malo (borroso, de perfil) no le quita el nombre a un track ya reconocido
                anterior = self.identity_cache.get(track_id)
                if nombre_detectado == 'Desconocido' and anterior and anterior[0] != 'Desconocido':
                    nombre_detectado, confianza_identidad = anterior
                self.identity_cache[track_id] = (nombre_detectado, confianza_identidad)
            else:
                nombre_detectado, confianza_identidad = self.identity_cache.get(
                    track_id, ('Desconocido', 0.0)
                )

            det_msg = FaceDetection()
            det_msg.track_id = int(track_id)
            det_msg.name = nombre_detectado
            det_msg.confidence = confianza_identidad
            det_msg.x1 = int(x1)
            det_msg.y1 = int(y1)
            det_msg.x2 = int(x2)
            det_msg.y2 = int(y2)
            self.detection_pub.publish(det_msg)

            color_caja = (0, 255, 0) if nombre_detectado != 'Desconocido' else (0, 0, 255)
            label = f"ID: {int(track_id)} | {nombre_detectado} ({confianza_identidad:.2f})"
            self.dibujar_caja(frame, (int(x1), int(y1), int(x2), int(y2)), label, color_caja)

            habla = self.ultimo_habla.get(track_id)
            if habla and time.time() - habla[2] < 4.0:
                texto = f"[{habla[1]}] {habla[0]}"
                texto = texto if len(texto) <= 45 else texto[:42] + '...'
                self.dibujar_texto(frame, texto, (int(x1), int(y2) + 18), (255, 200, 0))

        # limpia caché de tracks que ya no existen
        self.identity_cache = {t: v for t, v in self.identity_cache.items() if t in self.tracks}
        self.ultimo_habla = {t: v for t, v in self.ultimo_habla.items() if t in self.tracks}

        self.dibujar_texto(frame, f"Rostros: {len(boxes)}", (10, 20), (255, 255, 255))

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
        if success:
            msg_compressed = CompressedImage()
            msg_compressed.header = msg.header
            msg_compressed.header.frame_id = msg.header.frame_id or 'camera_frame'
            msg_compressed.format = "jpeg"
            msg_compressed.data = encoded_image.tobytes()
            self.annotated_pub.publish(msg_compressed)

    @staticmethod
    def dibujar_texto(frame, texto, org, color):
        (w, h), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = org
        y = max(y, h + base)
        cv2.rectangle(frame, (x, y - h - base), (x + w, y + base), (0, 0, 0), -1)
        cv2.putText(frame, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def dibujar_caja(self, frame, caja, label, color):
        x1, y1, x2, y2 = caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self.dibujar_texto(frame, label, (x1, y1 - 6), color)

    def assign_track_ids(self, boxes):
        assigned_ids = []
        used_track_ids = set()
        for box in boxes:
            best_iou, best_id = 0.0, None
            for tid, prev_box in self.tracks.items():
                score = iou(box, prev_box)
                if score > best_iou and score > self.iou_threshold and tid not in used_track_ids:
                    best_iou, best_id = score, tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used_track_ids.add(best_id)
            assigned_ids.append(best_id)
        self.tracks = {tid: box for tid, box in zip(assigned_ids, boxes)}
        return assigned_ids


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()