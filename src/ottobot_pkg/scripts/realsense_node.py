#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Int8
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError

from realsense_d435i import RealsenseD435i
from utils import quaternion_from_euler

import pyrealsense2 as rs
import numpy as np
import cv2
import yaml
from  pathlib import Path

# PyYaml workaround for indentation
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

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
        
        self.linear_acc_cov = rospy.get_param("~linear_acc_cov", 0.01)
        self.angular_vel_cov = rospy.get_param("~angular_vel_cov", 0.01)
        self.estimate_orientation = rospy.get_param("~estimate_orientation", False)

        self.enable_imu = rospy.get_param("~enable_imu", False)
        self.create_pc = rospy.get_param("~create_pc", False)

        self.calibration_filename = rospy.get_param("~calibration_filename", "camera_intrinsics")

        self.rgb_pub = rospy.Publisher(self.name+"/raw/rgb_image", 
                                        Image, 
                                        queue_size=5)
        self.depth_pub = rospy.Publisher(self.name+"/raw/depth_image", 
                                        Image, 
                                        queue_size=5)

        if self.enable_imu:
            self.imu_pub = rospy.Publisher(self.name+"/imu", 
                                            Imu, 
                                            queue_size=5)

        rospy.loginfo("Initializing camera node:\nenable_imu: {}\nestimate_orientation: {}\nfps: {}".format(
                        self.enable_imu, self.estimate_orientation, self.frame_rate
                    ))
        self.camera = None                                       
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev_id = dev.get_info(rs.camera_info.serial_number)
            rospy.loginfo("Device found '{}' ({})".format(dev.get_info(rs.camera_info.name), dev_id))
            rospy.loginfo("Resetting..")
            dev.hardware_reset()
            self.camera = RealsenseD435i(ctx, self.frame_rate,
                                        True, True, self.enable_imu,
                                        self.create_pc, dev_id)
    
    def run(self):

        bridge = CvBridge()

        # Prepare IMU message
        orientation_est = OrientationEstimator(0.98)

        imu_msg = Imu()
        imu_msg.header.frame_id = self.frame_id
        
        # Covariances
        imu_msg.linear_acceleration_covariance = [self.linear_acc_cov, 0.0, 0.0, 0.0, self.linear_acc_cov, 0.0, 0.0, 0.0, self.linear_acc_cov]
        imu_msg.angular_velocity_covariance = [self.angular_vel_cov, 0.0, 0.0, 0.0, self.angular_vel_cov, 0.0, 0.0, 0.0, self.angular_vel_cov]

        # Get camera intrinsics and dump them in configuration file
        camera_matrix, coeffs = self.camera.get_intrinsics()
        config_path = Path(__file__).resolve().parent.parent.joinpath("config")

        if config_path.exists() and config_path.is_dir():
            filename = config_path.joinpath(self.calibration_filename).with_suffix('.yaml')
            rospy.loginfo("Dumping camera intrinsics to '{}'..".format(str(filename)))
            data = {"camera_matrix": camera_matrix.tolist(), "coeffs":coeffs.tolist()}

            with open(filename, 'w') as f:
                yaml.dump(data, f, Dumper=MyDumper)
        else:
            rospy.logerror("Config directory does not exist: {}".format(str(config_path)))


        if not self.estimate_orientation:
            imu_msg.orientation_covariance = [-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            imu_msg.orientation.x = 0.0
            imu_msg.orientation.y = 0.0
            imu_msg.orientation.z = 0.0
            imu_msg.orientation.w = 0.0
        else:
            imu_msg.orientation_covariance = [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]

        try:
            while not rospy.is_shutdown():
                # Read camera
                color_image, depth_image, points, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = self.camera.poll()
                
                # Get time stamp
                now = rospy.get_rostime()

                if self.enable_imu:
                    # Prepare IMU msg
                    imu_msg.header.stamp = now
                    imu_msg.linear_acceleration.x = accel_x
                    imu_msg.linear_acceleration.y = accel_y
                    imu_msg.linear_acceleration.z = accel_z

                    # Considerinf differences in Realsense frame convention and ROS's one
                    imu_msg.angular_velocity.x = gyro_z
                    imu_msg.angular_velocity.y = gyro_x
                    imu_msg.angular_velocity.z = gyro_y

                    if self.estimate_orientation:
                        # Estimate IMU orientation
                        ts = frames.get_timestamp()
                        theta_x, theta_y, theta_z = orientation_est.get_orientation(accel, gyro, ts)
                        orientation = quaternion_from_euler(theta_z, theta_x, theta_y)
                        # orientation = quaternion_from_euler(theta_x, theta_y, theta_z)
                        imu_msg.orientation.x = orientation[0]
                        imu_msg.orientation.y = orientation[1]
                        imu_msg.orientation.z = orientation[2]
                        imu_msg.orientation.w = orientation[3]

                # Prepare image messages
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

                # Publish messages
                if self.enable_imu:
                    self.imu_pub.publish(imu_msg)
                self.rgb_pub.publish(color_image_msg)
                self.depth_pub.publish(depth_image_msg)

        finally:
            self.camera.shutdown()

if __name__ == "__main__":
    rs_camera_node = RealSenseCameraNode()
    try:
        rs_camera_node.run()
    except rospy.ROSInterruptionException:
        pass
    