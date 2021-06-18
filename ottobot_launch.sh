#!/usr/bin/env bash
source devel/setup.bash
export ROS_MASTER_URI=http://$HOSTNAME.local:11311
export ROS_HOSTNAME=$HOSTNAME

echo "Launching Ottobot..."
roslaunch ottobot_pkg camera_launch.launch