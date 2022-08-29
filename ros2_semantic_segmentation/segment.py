import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String

import cv2
from cv_bridge import CvBridge

from .run_deeplab import DeeplabInference


class SemanticSegmentation(Node):

    def __init__(self):
        super().__init__('semantic_segmentation')
        self.subscription = self.create_subscription(String, '/uav1/state', self.state_callback, 1)
        self.subscription = self.create_subscription(Image, '/uav1/slot3/image_raw', self.image_callback, 1)
        self.subscription  # prevent unused variable warning

        self.state = "IDLE"

        self.publisher = self.create_publisher(Image, '/segmentation_mask', 1)
        self.bridge = CvBridge()

        self.model_path = '/home/developer/mbzirc_ws/src/ros2_semantic_segmentation/models/scenario_model'
        self.deeplab_predict = DeeplabInference(self.model_path, ros_structure=True)
        self.get_logger().info('Model loaded')

        self. mask = cv2.imread('/home/developer/mbzirc_ws/src/ros2_semantic_segmentation/data/mask.png', cv2.IMREAD_GRAYSCALE)

    def state_callback(self, msg):
        self.state = msg.data

    def image_callback(self, msg):
        if self.state =="SEARCH" or self.state == "SERVO":
            self.get_logger().info('New msg')
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            img = img.astype("float32")

            # mask suction gripper
            img = cv2.bitwise_and(img, img, mask = self.mask)    
            img = img[:,80:560,:]

            # inference        
            mask = self.deeplab_predict.predict(img)
            mask = mask.astype(np.uint8)
            mask_msg = self.bridge.cv2_to_imgmsg(mask, 'rgb8')
            mask_msg.header = msg.header
            self.publisher.publish(mask_msg)
            self.get_logger().info('Published segmentation mask')


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = SemanticSegmentation()
    rclpy.spin(minimal_subscriber)

    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
