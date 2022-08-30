import rclpy
from rclpy.node import Node

from ros_ign_interfaces.msg import StringVec
from std_msgs.msg import String


class DummyPublisher(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')
        self.status_pub_ = self.create_publisher(String, '/mbzirc/target/stream/status', 1)
        self.report_sub_ = self.create_subscription(StringVec, 'target_report', self.report_callback, 1)
       
    def report_callback(self, msg):
        confirmation_msg = String()
        if msg.data[0] == 'small':
            confirmation_msg.data = 'small_object_id_success'
        elif msg.data[0] == 'large':
            confirmation_msg.data = 'large_object_id_success'
        elif msg.data[0] == 'vessel':
            confirmation_msg.data = 'vessel_id_success'
        self.status_pub_.publish(confirmation_msg)

def main(args=None):
    rclpy.init(args=args)

    dummy_publisher = DummyPublisher()
    rclpy.spin(dummy_publisher)

    dummy_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
