import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import time

from insightface.app import FaceAnalysis
from vision_interfaces.msg import FaceDetection

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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

class IdentityNode(Node):
    def __init__(self):
        self.frame_count = 0
        super().__init__('identity')
        
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.bridge = CvBridge()
        
        # Suscripciones
        self.image_sub = self.create_subscription(Image, 'camera/image_raw', self.image_callback, qos)
        self.det_sub = self.create_subscription(FaceDetection, 'vision/detections', self.detection_callback, 10)
        
        # Publicador de video anotado
        self.annotated_pub = self.create_publisher(Image, 'vision/image_annotated', 10)
        
        # Caché de detecciones de YOLO
        self.active_tracks = {}
        
        # Cargar InsightFace
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 480))
        self.get_logger().info('InsightFace cargado correctamente')
        
        # Enrolamiento
        self.known_embedding = None
        self.known_name = "Alejandro"
        self.identity_threshold = 0.40
        self.enroll_face('/home/usuario/ros2_ws/src/vision/alex.jpeg')

    def enroll_face(self, image_path):
        img = cv2.imread(image_path)
        if img is not None:
            faces = self.face_app.get(img)
            if len(faces) > 0:
                self.known_embedding = faces[0].embedding
                self.get_logger().info(f'Enrolamiento exitoso para {self.known_name}')
            else:
                self.get_logger().error('No se detectó cara en la foto de enrolamiento')
        else:
            self.get_logger().error('No se pudo cargar la imagen de enrolamiento')

    def detection_callback(self, msg):
        self.active_tracks[msg.track_id] = {
            'bbox': [msg.x1, msg.y1, msg.x2, msg.y2],
            'time': time.time()
        }

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return  # Ignora el frame y libera el procesador
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Limpiar tracks viejos
        current_time = time.time()
        self.active_tracks = {tid: data for tid, data in self.active_tracks.items() if current_time - data['time'] < 0.5}
        
        if self.active_tracks:
            # InsightFace 
            insight_faces = self.face_app.get(frame)
            
            for track_id, data in self.active_tracks.items():
                yolo_box = data['bbox']
                x1, y1, x2, y2 = yolo_box
                
                # Cruzar la caja de YOLO con las caras encontradas por InsightFace
                best_iou = 0.0
                best_face = None
                for iface in insight_faces:
                    score = iou(yolo_box, iface.bbox)
                    if score > best_iou:
                        best_iou = score
                        best_face = iface
                
                name = "Desconocido"
                conf = 0.0
                if best_face is not None and self.known_embedding is not None:
                    sim = cosine_similarity(best_face.embedding, self.known_embedding)
                    conf = float(sim)
                    if sim > self.identity_threshold:
                        name = self.known_name
                
                # Dibujar
                color = (0, 255, 0) if name == self.known_name else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID: {track_id} | {name} ({conf:.2f})"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Forzamos el formato a uint8 para que ROS lo entienda
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        
        # Creamos el mensaje ROS manualmente saltándonos cv_bridge
        from sensor_msgs.msg import Image
        annotated_msg = Image()
        annotated_msg.header = msg.header  # Reutilizamos el tiempo y frame_id original
        annotated_msg.height = frame.shape[0]
        annotated_msg.width = frame.shape[1]
        annotated_msg.encoding = 'bgr8'
        annotated_msg.is_bigendian = 0
        annotated_msg.step = frame.shape[1] * 3
        annotated_msg.data = frame.tobytes()
        
        self.annotated_pub.publish(annotated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IdentityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()