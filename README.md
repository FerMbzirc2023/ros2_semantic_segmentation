# ros2_semantic_segmentation
ROS 2 package for semantic segmentation.

## Installing dependencies
```
pip3 install opencv-python, tensorflow, scipy
sudo apt install ros-galactic-message-filters
cd /path/to/catkin_ws/src
git clone https://github.com/Box-Robotics/ros2_numpy 
cd ..
rosdep install -r --from-paths src -i -y --rosdistro galactic 
```

## Running the node
```
ros2 run ros2_semantic_segmentation mbzirc_semantic_segmentation 
```

## MBZIRC color codes

| ID    | Object                      | Color Code      |
| ------| ----------------------------| --------------- |
|1      | large ammo can              | (255,255,255)   |
|2      | large ammo can handles      | (255,0,0)       |
|3      | large crate                 | (0,255,0)       |
|4      | large crate handles         | (0,0,255)       |
|5      | large dry box               | (0, 255, 255)   |
|6      | large dry box handles       | (255, 0, 255)   |
|7      | large grey box              | (255, 255, 0)   |
|8      | large grey box handles      | (100, 100, 100) |
|9      | small blue box              | (100, 0, 0)     |
|10     | small case                  | (0, 100, 0)     |
|11     | small dry bag               | (0, 0, 100)     |
|12     | small dry bag handles       | (0, 100, 100)   |


## MBZIRC scenario color codes

| ID    | Object                      | Color Code      |
| ------| ----------------------------| --------------- |
|1      | large ammo can handles      | (255,255,255)   |
|2      | large crate handles         | (255,0,0)       |
|3      | large dry box handles       | (0,255,0)       |
|4      | small blue box              | (0,0,255)       |
|5      | small dry bag handles       | (0, 255, 255)   |
