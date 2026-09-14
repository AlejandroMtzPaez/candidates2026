import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # clave: minimiza el buffer

        if not self.cap.isOpened():
            self.get_logger().error('No se pudo abrir la cámara en el índice 0')

        timer_period = 1.0 / 15.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        t0 = time.time()
        ret, frame = self.cap.read()
        t1 = time.time()
        if not ret:
            self.get_logger().warn('No se pudo leer un frame de la cámara')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.publisher_.publish(msg)
        t2 = time.time()

        self.get_logger().info(f'read: {t1-t0:.3f}s | publish: {t2-t1:.3f}s')

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()