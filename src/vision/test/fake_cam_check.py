"""Prueba manual: publica una foto como cámara y verifica la salida de face_detector."""
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CompressedImage
from vision_interfaces.msg import FaceDetection

rclpy.init()
n = Node('fake_cam_check')
img = cv2.resize(cv2.imread(sys.argv[1]), (640, 480))
pub = n.create_publisher(Image, 'camera/image_raw', qos_profile_sensor_data)
dets, jpgs = [], []
n.create_subscription(FaceDetection, '/vision/detections', dets.append, 10)
n.create_subscription(CompressedImage, '/vision/image_annotated/compressed', jpgs.append, 10)
t0 = time.time()
while time.time() - t0 < 25 and len(jpgs) < 10:
    m = Image(height=480, width=640, step=1920, encoding='bgr8')
    m.data = img.tobytes()
    pub.publish(m)
    rclpy.spin_once(n, timeout_sec=0.1)
print('jpeg recibidos:', len(jpgs), 'formato:', jpgs[-1].format if jpgs else None)
print('detecciones:', sorted({(d.track_id, d.name, round(d.confidence, 2)) for d in dets}))
if jpgs:
    out = cv2.imdecode(np.frombuffer(bytes(jpgs[-1].data), np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite('/tmp/annotated.jpg', out)
    print('imagen anotada:', out.shape)
