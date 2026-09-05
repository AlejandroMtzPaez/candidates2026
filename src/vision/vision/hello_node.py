import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HelloNode(Node):
    def __init__(self):
        super().__init__('hello_node')
        self.publisher_ = self.create_publisher(String, 'saludo', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        self.counter += 1
        msg = String()
        msg.data = f'Hola desde ROS 2, tick #{self.counter}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publiqué: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    node = HelloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()