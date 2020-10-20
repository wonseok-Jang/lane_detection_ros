# lane_detector_ros

### Environment
* Jetson AGX Xavier
* CUDA 10.2
* CUDNN 8.0
* PyTorch 1.6.0
* torchvision 0.6.0
* ROS melodic
* Python 3.6.9

### Installation
* Clone repo in your workspace 
```
$ cd <CATKIN_WORKSPACE>/src
$ git clone https://github.com/wonseok-Jang/lane_detector_ros
```
* Build using python3
```
Modify /opt/ros/melodic/etc/catkin/profile.d/1.ros_python_vision.sh
```
from 
```
export ROS_PYTHON_VERSION=2
```
to
```
export ROS_PYTHON_VERSION=3
```
and then source the modified file.
```
$ source /opt/ros/melodic/etc/catkin/profile.d/1.ros_python_vision.sh
$ cd <CATKIN_WORKSPACE> && catkin_make -j <CPU cores>
```

### Set param (In `launch/lane_detector.launch`)
* `model_path`: Weights path

### Subscribed topic
* `/usb_cam/image_raw`: sensor_msgs::Image (You can change topic name in `launch/lane_detector.launch`)




