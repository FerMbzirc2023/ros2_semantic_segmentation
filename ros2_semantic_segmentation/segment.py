import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge


class SemanticSegmentation(Node):

    def __init__(self):
        super().__init__('semantic_segmentation')
        self.subscription = self.create_subscription(Image, '/usv/arm/wrist/image_raw', self.image_callback, 1)
        self.subscription  # prevent unused variable warning

        self.publisher = self.create_publisher(Image, '/segmentation_mask', 1)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        self.get_logger().info('New msg')

        img = self.bridge.imgmsg_to_cv2(msg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # cv2.imshow('image', img)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows() 
        
        mask_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.publisher.publish(mask_msg)
        self.get_logger().info('Published')


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = SemanticSegmentation()
    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
