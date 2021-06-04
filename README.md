# ottobot
ROS based autonomous mobile robot

## catkin_workspace and cv_bridge support with Python3 and OpenCV 4.1.1 (Jetson Nano)

Create Catkin workspace

```shell
mkdir -p ~/catkin_ws/src -DPYTHON_EXECUTABLE=/usr/bin/python3
cd ~/catkin_ws
```

Download `cv_bridge` package for `noetic`. Although the Nano comes with Ubuntu 18.04(i.e. `melodic`), `noetic` is the one with support for installed OpenCv 4 (version installed by default on the Nano).

```shell
cd src
git clone -b noetic https://github.com/ros-perception/vision_opencv.git
```

Then, we need to modify line 11 in `~/catkin_ws/src/vision_opencv/cv_bridge/CMakeLists.txt` from:

`find_package(Boost REQUIRED python37)`

to 

`find_package(Boost REQUIRED python3)`

Finally we build the package

```shell
cd ~/catkin_ws
catkin_make
```