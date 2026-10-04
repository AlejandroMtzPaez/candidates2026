import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
#from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ultralytics import YOLO
import cv2
import numpy as np
from vision_interfaces.msg import FaceDetection
import pickle
import os
from insightface.app import FaceAnalysis
from numpy.linalg import norm
from sensor_msgs.msg import Image, CompressedImage

def similitud_coseno(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

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

        #self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, qos
        )
        self.detection_pub = self.create_publisher(
            FaceDetection, 'vision/detections', 10
        )
        #self.annotated_pub = self.create_publisher(
            #Image, 'vision/image_annotated', 10
        #)

        self.annotated_pub = self.create_publisher(CompressedImage, '/vision/image_annotated/compressed', 10)


        model_path = self.declare_parameter(
            'model_path',
            '/home/usuario/ros2_ws/src/vision/models/yolov8n-face.pt'
        ).value

        self.model = YOLO(model_path)
        self.get_logger().info('Modelo YOLOv8-face cargado correctamente')

        # carga de identidades 
        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.db_embeddings = {}
        db_path = os.path.expanduser('~/ros2_ws/src/vision/vision/caras_enroladas.pkl')
        if os.path.exists(db_path):
            with open(db_path, 'rb') as f:
                self.db_embeddings = pickle.load(f)
            self.get_logger().info(f"Identidades cargadas: {list(self.db_embeddings.keys())}")
        self.umbral_identidad = 0.3 
        
        
        # Tracking simple por IoU
        self.tracks = {}      # track_id -> bbox
        self.next_id = 0
        self.iou_threshold = 0.3

    def image_callback(self, msg):
        # DESEMPAQUETADO MANUAL
        # Convertimos los bytes crudos de ROS 2 directo a una matriz de imagen
        frame = np.ndarray(shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data)
        
        # YOLO (Cajas y Tracking)
        results = self.model.predict(frame, verbose=False, conf=0.4)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
        current_frame_ids = self.assign_track_ids(boxes)

        #INSIGHTFACE
        insight_faces = self.face_app.get(frame)

        for box, track_id in zip(boxes, current_frame_ids):
            x1, y1, x2, y2 = box
            nombre_detectado = 'Desconocido'
            confianza_identidad = 0.0
            
            # Calculamos el centroide (X, Y) de la caja de YOLO
            cx_yolo = (x1 + x2) / 2
            cy_yolo = (y1 + y2) / 2
            
            mejor_iface = None
            menor_distancia = float('inf')
            
            # Buscamos qué cara de InsightFace está más cerca de este centro
            for iface in insight_faces:
                ix1, iy1, ix2, iy2 = iface.bbox
                cx_iface = (ix1 + ix2) / 2
                cy_iface = (iy1 + iy2) / 2
                
                # Distancia euclidiana
                distancia = ((cx_yolo - cx_iface)**2 + (cy_yolo - cy_iface)**2)**0.5
                
                if distancia < menor_distancia:
                    menor_distancia = distancia
                    mejor_iface = iface

            # Si la distancia es menor a 150 pixeles, asumimos que es la misma cara
            if mejor_iface is not None and menor_distancia < 150:
                embedding_actual = mejor_iface.embedding
                mejor_score = 0.0
                
                for nombre_db, emb_db in self.db_embeddings.items():
                    score = similitud_coseno(embedding_actual, emb_db)
                    
                    if score > mejor_score:
                        mejor_score = score
                        # Aplicamos el umbral (0.40 o el que tengas configurado)
                        nombre_detectado = nombre_db if score >= self.umbral_identidad else 'Desconocido'
                
                confianza_identidad = float(mejor_score)

            # PUBLICAR MENSAJE DE IDENTIDAD 
            det_msg = FaceDetection()
            det_msg.track_id = int(track_id)
            det_msg.name = nombre_detectado
            det_msg.confidence = confianza_identidad 
            det_msg.x1 = int(x1)
            det_msg.y1 = int(y1)
            det_msg.x2 = int(x2)
            det_msg.y2 = int(y2)
            self.detection_pub.publish(det_msg)

            # DIBUJAR EN EL FRAME
            # Verde si te reconoce, Rojo si es Desconocido
            color_caja = (0, 255, 0) if nombre_detectado != 'Desconocido' else (0, 0, 255)
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color_caja, 2)
            
            label = f"ID: {int(track_id)} | {nombre_detectado} ({confianza_identidad:.2f})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_caja, 2)

        """
        # --- 6. EMPAQUETADO MANUAL PARA PUBLICAR EL VIDEO ANOTADO ---
        annotated_msg = Image()
        annotated_msg.header = msg.header  # Reutilizamos el header original para mantener la sincronía
        annotated_msg.height = frame.shape[0]
        annotated_msg.width = frame.shape[1]
        annotated_msg.encoding = 'bgr8'
        annotated_msg.is_bigendian = 0
        annotated_msg.step = frame.shape[1] * 3
        annotated_msg.data = frame.tobytes()
        
        self.annotated_pub.publish(annotated_msg)
        """
        
        # --- 6. COMPRESIÓN JPEG PARA FOXGLOVE ---
        # Comprimimos la imagen al 60% de calidad para ahorrar red
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        success, encoded_image = cv2.imencode('.jpg', frame, encode_param)
        
        if success:
            msg_compressed = CompressedImage()
            msg_compressed.header = msg.header
            msg_compressed.format = "jpeg"
            msg_compressed.data = encoded_image.tobytes()
            
            self.annotated_pub.publish(msg_compressed)


    def assign_track_ids(self, boxes):
        assigned_ids = []
        used_track_ids = set()

        for box in boxes:
            best_iou = 0.0
            best_id = None
            for tid, prev_box in self.tracks.items():
                score = iou(box, prev_box)
                if score > best_iou and score > self.iou_threshold and tid not in used_track_ids:
                    best_iou = score
                    best_id = tid

            if best_id is None:
                best_id = self.next_id
                self.next_id += 1

            used_track_ids.add(best_id)
            assigned_ids.append(best_id)

        # Actualiza el diccionario de tracks solo con lo visto en este frame
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
