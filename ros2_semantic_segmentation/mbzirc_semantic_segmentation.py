import rclpy
from rclpy.node import Node

import time

import math
import numpy as np
import ros2_numpy as rnp
import message_filters

from sensor_msgs.msg import Image, PointCloud2
from ros_ign_interfaces.msg import StringVec
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from mbzirc_aerial_manipulation_msgs.srv import ChangeState

import cv2
from cv_bridge import CvBridge

from .run_deeplab import DeeplabInference
from .centroid_tracker import CentroidTracker


class SemanticSegmentation(Node):
    def __init__(self):
        super().__init__('semantic_segmentation')
        self.state_sub_ = self.create_subscription(String, 'state', self.state_callback, 1)
        self.status_sub_ = self.create_subscription(String, '/mbzirc/target/stream/status', self.status_callback, 1)
        
        self.sync_image_sub_ = message_filters.Subscriber(self, Image, 'slot3/image_raw')
        self.sync_pc_sub_ = message_filters.Subscriber(self, PointCloud2, 'slot3/points')
        self.ts = message_filters.TimeSynchronizer([self.sync_image_sub_, self.sync_pc_sub_], 1)
        self.ts.registerCallback(self.sync_callback)

        self.seg_mask_pub_ = self.create_publisher(Image, '/semantic_segentation/segmentation_mask', 1)
        self.centroid_img_pub_ = self.create_publisher(Image, '/semantic_segentation/detected_centroids', 1)
        self.report_pub_ = self.create_publisher(StringVec, 'target_report', 1)
        self.centroid_pub_ = self.create_publisher(PointStamped, 'semantic_segentation/detected_point', 1)

        self.change_state_client_ = self.create_client(ChangeState, "change_state")
        
        self.bridge = CvBridge()

        self.model_path = '/home/developer/mbzirc_ws/src/ros2_semantic_segmentation/models/scenario_model_shadows_basic'
        self.deeplab_predict = DeeplabInference(self.model_path, ros_structure=True)
        self.get_logger().info('Model loaded')

        self.gripper_mask = cv2.imread('/home/developer/mbzirc_ws/src/ros2_semantic_segmentation/data/mask.png', cv2.IMREAD_GRAYSCALE)
        
        self.state = "SEARCH"   # testing, set to IDLE!!
        self.small_target_identified = False
        self.targets_identified = False
        self.waiting_for_response = False

        self.small_target_id = 0
        self.large_target_id = 0

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

        self.large_object_timeout = 10

    def state_callback(self, msg):
        self.get_logger().info("New state: {}".format(msg.data))
        self.state = msg.data

        if self.state == 'SEARCH':
            self.small_target_identified = False
            self.targets_identified = False

        elif self.state == 'SERVOING':
            self.target_tracker = CentroidTracker()
            self.target_tracker.update([self.small_target_centroid])

    def status_callback(self, msg):
        print("Recieved target report status: " + msg.data)
        self.waiting_for_response = False
        if msg.data == 'small_object_id_success':
            self.small_target_identified = True
            self.start_time = time.time()
        elif msg.data == 'large_object_id_success':
            self.targets_identified = True
            req = ChangeState.Request()
            req.state = 'SERVOING'
            self.change_state_client_.call_async(req)
            # Check spin until future complete https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Service-And-Client.html

    def sync_callback(self, msg, pc_msg):
        #self.get_logger().info("Entered sync callback!")

        if self.state =="SEARCH" or self.state == 'SERVOING':
            #self.get_logger().debug("Entered!".format(msg.data))
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            img = img.astype("float32")

            # mask suction gripper
            img = cv2.bitwise_and(img, img, mask = self.gripper_mask)    

            # inference        
            self.seg_mask = self.deeplab_predict.predict(img)
            self.seg_mask = self.seg_mask.astype(np.uint8)
            mask_msg = self.bridge.cv2_to_imgmsg(self.seg_mask, 'rgb8')
            mask_msg.header = msg.header
            self.seg_mask_pub_.publish(mask_msg)

            #if self.state == 'SEARCH' and not self.targets_identified:
            if self.state == 'SEARCH' and not self.targets_identified:
                # track centroids
                for i in range(5):
                    mask = np.where(np.all(self.seg_mask == self.color_codes[i+1], axis=-1, keepdims=True), [255, 255, 255], [0, 0, 0])
                    mask = mask[:,:,0].astype(np.uint8)

                    # find contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    max_cnt = []
                    centroids = []

                    if len(contours) != 0:
                        c_sorted = sorted(contours, key=cv2.contourArea)
                        max_cnt.append(c_sorted[-1])
                        if len(contours) >= 2:
                            max_cnt.append(c_sorted[-2])
                    
                    for c in max_cnt:
                        if cv2.contourArea(c) > 30:
                            moments = cv2.moments(c)
                            if moments["m00"] > 0:
                                cX = int(moments["m10"] / moments["m00"])
                                cY = int(moments["m01"] / moments["m00"])	
        
                                centroids.append((cX,cY))

                    if len(centroids) != 0:
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
                            self.small_target_id = i+1

                            print("Reporting object " + self.object_ids[i+1])
                            report = StringVec()
                            report.data = ['small', str(int(self.small_target_centroid[0])), str(int(self.small_target_centroid[1]))]
                            self.report_pub_.publish(report)
                            self.waiting_for_response = True

                if not self.waiting_for_response and self.small_target_identified:
                    for i in range(3):
                        if self.trackers[i].confirmed:
                            self.large_target_centroid = self.trackers[i].confirmedCentroid
                            self.large_target_id = i+1

                            print("Reporting object " + self.object_ids[i+1])
                            report = StringVec()
                            report.data = ['large', str(int(self.large_target_centroid[0])), str(int(self.large_target_centroid[1]))]
                            self.report_pub_.publish(report)
                            #self.waiting_for_response = True
                    
                    # timeout occured
                    time_delta = time.time() - self.start_time
                    print(time_delta)
                    if time_delta > self.large_object_timeout:
                        self.get_logger().info("Large object not identified, moving to SERVOING")
                        req = ChangeState.Request()
                        req.state = 'SERVOING'
                        self.change_state_client_.call_async(req)


            if self.state == 'SERVOING':
                mask = np.where(np.all(self.seg_mask == self.color_codes[self.small_target_id], axis=-1, keepdims=True), [255, 255, 255], [0, 0, 0])
                mask = mask[:,:,0].astype(np.uint8)

                # find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                centroids = []


                if len(contours) != 0:
                    # cv2.drawContours(self.seg_mask, contours, -1, (255,255,255), 3)

                    for c in contours:
                        if cv2.contourArea(c) > 30:
                            # x,y,w,h = cv2.boundingRect(c)
                            # cv2.rectangle(self.seg_mask,(x,y),(x+w,y+h),(0,255,0),2)

                            moments = cv2.moments(c)
                            if moments["m00"] > 0:
                                cX = int(moments["m10"] / moments["m00"])
                                cY = int(moments["m01"] / moments["m00"])	

                                centroids.append((cX,cY))
                        else: 
                            self.get_logger().warn("Contours of an object too small!")

                else: 
                    self.get_logger().warn("No centroids found!")


                if len(centroids) != 0:
                    objects = self.target_tracker.update(centroids)
                    target_object_id = min(objects.keys())
                    self.small_target_centroid = objects[target_object_id]

                    text = "target object"
                    cv2.putText(self.seg_mask, text, (self.small_target_centroid[0] - 10, self.small_target_centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.circle(self.seg_mask, (self.small_target_centroid[0], self.small_target_centroid[1]), 4, (255, 255, 255), -1)

                    pc_array = rnp.point_cloud2.pointcloud2_to_array(pc_msg)

                    (x,y,z,_) = pc_array[self.small_target_centroid[1], self.small_target_centroid[0]]
                    if not math.isinf(x):
                        point_msg = PointStamped()
                        point_msg.header = pc_msg.header
                        point_msg.point.x = float(x)
                        point_msg.point.y = float(y)
                        point_msg.point.z = float(z)
                        self.centroid_pub_.publish(point_msg)
                    
                    else: 
                        self.get_logger().warn("Too far from an object!".format(msg.data))


            mask_msg = self.bridge.cv2_to_imgmsg(self.seg_mask, 'rgb8')
            self.centroid_img_pub_.publish(mask_msg)


def main(args=None):
    rclpy.init(args=args)

    semantic_segmentation = SemanticSegmentation()
    rclpy.spin(semantic_segmentation)

    semantic_segmentation.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
