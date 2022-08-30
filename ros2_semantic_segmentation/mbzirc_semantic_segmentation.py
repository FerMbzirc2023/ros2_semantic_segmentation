import rclpy
from rclpy.node import Node

import numpy as np
# import ros2_numpy as rnp

from sensor_msgs.msg import Image, PointCloud2
from ros_ign_interfaces.msg import StringVec
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

import cv2
from cv_bridge import CvBridge

from .run_deeplab import DeeplabInference
from .centroid_tracker import CentroidTracker


class SemanticSegmentation(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')
        self.state_sub_ = self.create_subscription(String, '/uav1/state', self.state_callback, 1)
        self.status_sub_ = self.create_subscription(String, '/mbzirc/target/stream/status', self.status_callback, 1)
        self.image_sub_ = self.create_subscription(Image, '/uav1/slot3/image_raw', self.image_callback, 1) # TODO: message filters for time synchronization

        self.state_pub_ = self.create_publisher(String, '/uav1/state', 1)
        self.seg_mask_pub_ = self.create_publisher(Image, '/segmentation_mask', 1)
        self.centroid_img_pub_ = self.create_publisher(Image, '/centroid_tracker/detected_centroids', 1)
        self.report_pub_ = self.create_publisher(StringVec, '/uav1/target_report', 1)
        self.centroid_pub_ = self.create_publisher(PointStamped, '/centroid_tracker/detected_point', 1)
        
        self.bridge = CvBridge()

        self.model_path = '/home/developer/mbzirc_ws/src/ros2_semantic_segmentation/models/scenario_model_brightness'
        self.deeplab_predict = DeeplabInference(self.model_path, ros_structure=True)
        self.get_logger().info('Model loaded')

        self.gripper_mask = cv2.imread('/home/developer/mbzirc_ws/src/ros2_semantic_segmentation/data/mask.png', cv2.IMREAD_GRAYSCALE)
        
        self.state = "SEARCH"   # testing, set to IDLE!!
        self.counter = 0
        self.small_target_identified = False
        self.targets_identified = False
        self.waiting_for_response = False

        self.color_codes =	{
            1: (255,255,255),
            2: (255,0,0),
            3: (0,255,0),
            4: (0,0,255),
            5: (0, 255, 255),
        }

        self.object_ids =	{
            1: 'LargeAmmoCanHandles',
            2: 'LargeCrateHandles',
            3: 'LargeDryBoxHandles',
            4: 'SmallBlueBox',
            5: 'SmallDryBagHandle',
        }
        self.trackers = [CentroidTracker() for i in range(5)]

    def state_callback(self, msg):
        print("New state: " + msg.data)
        self.state = msg.data

    def status_callback(self, msg):
        print("Recieved target report status: " + msg.data)
        self.waiting_for_response = False
        if msg.data == 'small_object_id_success':
            self.small_target_identified = True
        elif msg.data == 'large_object_id_success':
            self.targets_identified = True
            state_msg = String()
            state_msg.data = 'SERVO'
            self.state_pub_.publish(state_msg)

    def image_callback(self, msg):
        if self.state =="SEARCH" or self.state == "SERVO":
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            img = img.astype("float32")

            # mask suction gripper
            img = cv2.bitwise_and(img, img, mask = self.gripper_mask)    
            img = img[:,80:560,:]

            # inference        
            self.seg_mask = self.deeplab_predict.predict(img)
            self.seg_mask = self.seg_mask.astype(np.uint8)
            mask_msg = self.bridge.cv2_to_imgmsg(self.seg_mask, 'rgb8')
            mask_msg.header = msg.header
            self.seg_mask_pub_.publish(mask_msg)

            self.counter += 1

            # track centroids
            if self.state == 'SEARCH':
                if not self.targets_identified:
                    for i in range(5):
                        mask = np.where(np.all(self.seg_mask == self.color_codes[i+1], axis=-1, keepdims=True), [255, 255, 255], [0, 0, 0])
                        mask = mask[:,:,0].astype(np.uint8)

                        # find contours
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        max_cnt = []
                        centroids = []

                        if len(contours) != 0:
                            # cv2.drawContours(self.seg_mask, contours, -1, (255,255,255), 3)

                            c_sorted = sorted(contours, key=cv2.contourArea)
                            max_cnt.append(c_sorted[-1])
                            if len(contours) >= 2:
                                max_cnt.append(c_sorted[-2])
                        
                        for c in max_cnt:
                            if cv2.contourArea(c) > 30:
                                x,y,w,h = cv2.boundingRect(c)
                                # cv2.rectangle(self.seg_mask,(x,y),(x+w,y+h),(0,255,0),2)

                                moments = cv2.moments(c)
                                if moments["m00"] > 0:
                                    cX = int(moments["m10"] / moments["m00"])
                                    cY = int(moments["m01"] / moments["m00"])	
            
                                    centroids.append((cX,cY))

                        objects = self.trackers[i].update(centroids)
                        for (objectID, centroid) in objects.items():
                            text = "ID {}".format(objectID)
                            cv2.putText(self.seg_mask, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            text = self.object_ids[i+1]
                            cv2.putText(self.seg_mask, text, (centroid[0] - 20, centroid[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            cv2.circle(self.seg_mask, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

                    if not self.waiting_for_response and not self.small_target_identified:
                        for i in range(3,5):
                            if self.trackers[i].confirmed:
                                self.small_target_centroid = self.trackers[i].confirmedCentroid

                                print("Reporting object " + self.object_ids[i+1])
                                cv2.circle(self.seg_mask, (self.small_target_centroid[0], self.small_target_centroid[1]), 7, (0, 255, 0), -1)

                                report = StringVec()
                                report.data = ['small', str(int(self.small_target_centroid[0])), str(int(self.small_target_centroid[1]))]
                                self.report_pub_.publish(report)
                                self.waiting_for_response = True

                    if not self.waiting_for_response and self.small_target_identified:
                        for i in range(3):
                            if self.trackers[i].confirmed:
                                self.large_target_centroid = self.trackers[i].confirmedCentroid

                                print("Reporting object " + self.object_ids[i+1])
                                cv2.circle(self.seg_mask, (self.small_target_centroid[0], self.small_target_centroid[1]), 7, (0, 255, 0), -1)

                                report = StringVec()
                                report.data = ['large', str(int(self.large_target_centroid[0])), str(int(self.large_target_centroid[1]))]
                                self.report_pub_.publish(report)
                                self.waiting_for_response = True

                    mask_msg = self.bridge.cv2_to_imgmsg(self.seg_mask, 'rgb8')
                    self.centroid_img_pub_.publish(mask_msg)

            elif self.state == 'SERVO':
                pass
                # pocetni centroid imas, od novih detektiranih velicine iznad nekog delta, objavi poziciju onog blizeg prethodnom

    def pointcloud_callback():
        pass


def main(args=None):
    rclpy.init(args=args)

    semantic_segmentation = SemanticSegmentation()
    rclpy.spin(semantic_segmentation)

    semantic_segmentation.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
