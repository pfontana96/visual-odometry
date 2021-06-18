#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Int8
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros

import pyrealsense2 as rs
import numpy as np
import cv2

def config_camera(fps, accel_fps, gyro_fps):
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

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, fps)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, accel_fps)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_fps)

    return pipeline, config

class OrientationEstimator(object):
    def __init__(self, alpha):
        self.alpha = alpha

        self.first_frame = True
        self.last_ts = None

        # Orientation angles
        self.theta_x = 0
        self.theta_y = 0
        self.theta_z = 0
    

    def get_orientation(self, accel_data, gyro_data, ts):
        # Process accelerometer
        accel_angle_z = np.arctan2(accel_data.y, accel_data.z)
        accel_angle_x = np.arctan2(accel_data.x, np.sqrt(accel_data.y**2 + accel_data.z**2))

        if self.first_frame:
            
            self.first_frame = False

            # We cannot infer angle around Y axis using only accelerometer, by convention we set 
            # initial pose to PI
            accel_angle_y = np.pi

            self.last_ts = ts
            self.theta_x = accel_angle_x
            self.theta_y = accel_angle_y
            self.theta_z = accel_angle_z

            return (self.theta_x, self.theta_y, self.theta_z)

        # Process gyroscope
        dt_gyro = ts - self.last_ts
        dangle_x = gyro_data.x * dt_gyro
        dangle_y = gyro_data.y * dt_gyro
        dangle_z = gyro_data.z * dt_gyro

        gyro_angle_x = self.theta_x + dangle_x
        gyro_angle_y = self.theta_y + dangle_y
        gyro_angle_z = self.theta_z + dangle_z

        self.last_ts = ts
        # Fuse sensor's data
        self.theta_x = gyro_angle_x*self.alpha + accel_angle_x*(1-self.alpha)
        self.theta_z = gyro_angle_z*self.alpha + accel_angle_z*(1-self.alpha)
        self.theta_y = gyro_angle_y

        return (self.theta_x, self.theta_y, self.theta_z)

class RealSenseCameraNode(object):
    def __init__(self):
        self.name = rospy.get_param("name", "realsense_camera")
        rospy.init_node(self.name)

        self.frame_id = rospy.get_param("~frame_id", "map")
        self.frame_rate = rospy.get_param("~frame_rate", 15) # Defaults to 15 fps
        
        self.rgb_pub = rospy.Publisher(self.name+"/raw/rgb_image", 
                                        Image, 
                                        queue_size=5)
        self.depth_pub = rospy.Publisher(self.name+"/raw/depth_image", 
                                        Image, 
                                        queue_size=5)
        self.imu_pub = rospy.Publisher(self.name+"/imu", 
                                        Imu, 
                                        queue_size=5)
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()

        self.pipeline, self.config = config_camera(self.frame_rate, 250, 200)
    
    def run(self):

        # Start streaming
        profile = self.pipeline.start(self.config)
        rospy.loginfo("Started streaming..")

        bridge = CvBridge()

        # Prepare IMU message
        orientation_est = OrientationEstimator(0.98)

        imu_msg = Imu()
        imu_msg.header.frame_id = self.frame_id

        try:
            while not rospy.is_shutdown():
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()
                # Get ROS time (this is NOT actually capturing time)
                now = rospy.get_rostime()

                # # frames.get_depth_frame() is a 640x360 depth image
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                # Get IMU data (changes a bit from the rest of the API)
                ctn = 0
                for frame in frames:
                    if frame.is_motion_frame():
                        if not ctn: # accel data
                            accel = frame.as_motion_frame().get_motion_data()
                        elif ctn == 1: # gyro data
                            gyro = frame.as_motion_frame().get_motion_data()
                        ctn += 1

                # Prepare IMU msg
                imu_msg.header.stamp = now
                imu_msg.linear_acceleration.x = accel.x
                imu_msg.linear_acceleration.y = accel.y
                imu_msg.linear_acceleration.z = accel.z

                imu_msg.angular_velocity.x = gyro.x
                imu_msg.angular_velocity.y = gyro.y
                imu_msg.angular_velocity.z = gyro.z

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Convert images to ROS messages
                try:
                    color_image_msg = bridge.cv2_to_imgmsg(color_image.astype(np.int8), encoding="passthrough")
                    depth_image_msg = bridge.cv2_to_imgmsg(depth_image.astype(np.int8), encoding="passthrough")
                except CvBridgeError as e:
                    rospy.logerr(e)
                    continue

                color_image_msg.header.stamp = now
                depth_image_msg.header.stamp = now

                color_image_msg.header.frame_id = self.frame_id
                color_image_msg.header.frame_id = self.frame_id

                # Estimate IMU orientation
                ts = frames.get_timestamp()
                theta_x, theta_y, theta_z = orientation_est.get_orientation(accel, gyro, ts)
                orientation = quaternion_from_euler(theta_x, theta_y, theta_z)
                imu_msg.orientation.x = orientation[0]
                imu_msg.orientation.y = orientation[1]
                imu_msg.orientation.z = orientation[2]
                imu_msg.orientation.w = orientation[3]

                # Publish images
                self.imu_pub.publish(imu_msg)
                self.rgb_pub.publish(color_image_msg)
                self.depth_pub.publish(depth_image_msg)

        finally:
            self.pipeline.stop()

if __name__ == "__main__":
    rs_camera_node = RealSenseCameraNode()
    rs_camera_node.run()
    
