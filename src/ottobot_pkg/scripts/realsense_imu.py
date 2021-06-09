#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Int8
from sensor_msgs.msg import Imu

import pyrealsense2 as rs
import numpy as np

def config_IMU(accel_fps, gyro_fps):
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_fps)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_fps)
    
    return pipeline, config

class RealSenseIMUNode(object):
    def __init__(self):
        self.name = rospy.get_param("name", "realsense_imu")
        rospy.init_node(self.name)

        self.frame_id = rospy.get_param("frame_id", "map")
        self.accel_frame_rate = 250
        self.gyro_frame_rate = 200

        self.imu_pub = rospy.Publisher(self.name+"/imu", 
                                        Imu, 
                                        queue_size=10)

        self.pipeline, self.config = config_IMU(self.accel_frame_rate, self.gyro_frame_rate)
    
    def run(self):

        # Start streaming
        profile = self.pipeline.start(self.config)
        rospy.loginfo("Started streaming..")

        # Prepare message
        imu_msg = Imu()

        imu_msg.header.frame_id = self.frame_id

        imu_msg.orientation.x = 0
        imu_msg.orientation.y = 0
        imu_msg.orientation.z = 0
        imu_msg.orientation.w = 0

        try:
            while not rospy.is_shutdown():
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()

                # Get ROS time (this is NOT actually capturing time)
                now = rospy.get_rostime()

                accel = frames[0].as_motion_frame().get_motion_data()
                gyro = frames[1].as_motion_frame().get_motion_data()

                # Publish IMU data
                imu_msg.header.stamp = now
                imu_msg.linear_acceleration.x = accel.x
                imu_msg.linear_acceleration.y = accel.y
                imu_msg.linear_acceleration.z = accel.z

                imu_msg.angular_velocity.x = gyro.x
                imu_msg.angular_velocity.y = gyro.y
                imu_msg.angular_velocity.z = gyro.z

                self.imu_pub.publish(imu_msg)

        finally:
            self.pipeline.stop()

if __name__ == "__main__":
    rs_camera_node = RealSenseIMUNode()
    rs_camera_node.run()
