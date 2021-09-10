#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Int8
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError

from utils import quaternion_from_aa

import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy

ARUCO_TARGET_ID = 2 # Target ArUco ID

@dataclass
class ArucoMarker(object):
    """
    Data class for detected ARUCO marker:

    Attributes:
    id: (int) Aruco ID
    corners: (np.ndarray) a 4x2 array containing (top_left, top_right, bottom_right, bottom_left)
              coordinates
    trans_vec: (np.ndarray) array containing (x, y, z) of detected marker
    rot_mat: (np.ndarray) array containing the axis-angle representation of the rotation
    """
    id: int
    corners: np.ndarray
    trans_vec: np.ndarray
    rot_vec: np.ndarray

    def __init__(self, id: int, corners: np.ndarray, trans_vec: np.ndarray, rot_vec: np.ndarray):
        assert corners.shape == (4,2), "corners shape should be (4,2), got {} instead".format(corners.shape)
        assert trans_vec.shape == (3,), "trans_vec shape should be (3), got {} instead".format(trans_vec.shape)
        assert rot_vec.shape == (3,), "rot_vec shape should be (3,), got {} instead".format(rot_vec.shape)
        
        self.id = id
        self.corners = corners 
        self.t = trans_vec
        self.r = rot_vec

    def get_pose(self):
        return self.t, quaternion_from_aa(self.r)

    def draw_marker(self, image, camera_matrix=None, coeffs=None):
        image = cv2.drawContours(image, [self.corners], 0, (36,255,12), 2)

        if (camera_matrix is not None) and (coeffs is not None):
            image = cv2.aruco.drawAxis(image, camera_matrix, coeffs, self.r, self.t, 0.1)
        
        return image

class QRDetectorNode(object):
    def __init__(self, debug=False):
        self.name = rospy.get_param("~name", "qr_detector")
        if debug:
            log_level = rospy.INFO
        else:
            log_level = rospy.DEBUG

        rospy.init_node(self.name, log_level=log_level)

        img_in = rospy.get_param("~img_in", self.name + "/img_in")
        img_out = rospy.get_param("~img_out", self.name + "/img_out")
        pose_out = rospy.get_param("~pose_out", self.name + "/pose_out")

        frame_rate = rospy.get_param("~frame_rate", 30)
        filename = rospy.get_param("~calib_filename", "camera_intrinsics.yaml")

        self.img_sub = rospy.Subscriber(img_in, Image, self.callback)
        self.img_pub = rospy.Publisher(img_out, Image, queue_size=5)
        self.pose_pub = rospy.Publisher(pose_out, PoseStamped, queue_size=5)
        
        self.bridge = CvBridge()
        self.image = None
        self.stamp = None
        self.frame_id = None

        self.detected_marker = None

        self.rate = rospy.Rate(frame_rate) # Hz

        # Using OpenCV and ArUco codes 
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Load camera intrinsics
        self.camera_matrix = None
        self.coeffs = None
        calib_filename = Path(__file__).resolve().parent.parent.joinpath("config/" + filename)
        try:
            with open(calib_filename, 'r') as f:
                data = yaml.load(f, Loader=yaml.loader.SafeLoader)
            self.camera_matrix = np.array(data['camera_matrix'])
            self.coeffs = np.array(data['coeffs'])
        except Exception as e:
            rospy.logerr(e)

    
    def callback(self, data):
        # Convert ROS msg to cv2
        try:
           self.image = self.bridge.imgmsg_to_cv2(data, "passthrough").astype(np.uint8)
           self.stamp = data.header.stamp
           self.frame_id = data.header.frame_id
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Using ArUco
        corners, ids, rejected = cv2.aruco.detectMarkers(self.image, self.aruco_dict, 
                                                        parameters=self.aruco_params)
        
        # At least one code detected
        if len(corners) > 0:
            ids = ids.flatten()
            
            # Loop over detected markers
            for (m_corners, m_id) in zip(corners, ids):
                # rospy.loginfo("Marker detected ({}): {}".format(m_id, m_corners))
                
                # If detected marker is target
                if m_id == ARUCO_TARGET_ID:

                    if (self.camera_matrix is not None) and (self.coeffs is not None):
                        # Estimate marker pose
                        rot_vec, trans_vec = cv2.aruco.estimatePoseSingleMarkers(m_corners, 0.165, self.camera_matrix, self.coeffs)
                        # rospy.loginfo("Rotation vec:\n{}\nTranslation vec:\n{}".format(rot_vec, trans_vec))
                        self.detected_marker = ArucoMarker(ARUCO_TARGET_ID, 
                                                           m_corners.reshape((4,2)).astype(int), 
                                                           trans_vec.reshape(-1), 
                                                           rot_vec.reshape(-1))

    def run(self):
        while not rospy.is_shutdown():
            if self.image is not None:
                image = deepcopy(self.image)
                if self.detected_marker is not None:
                    image = self.detected_marker.draw_marker(image,
                                                             self.camera_matrix,
                                                             self.coeffs)
                # Prepare ROS message (modified image)
                try:
                    image = self.bridge.cv2_to_imgmsg(image.astype(np.int8), encoding="passthrough")
                    image.header.stamp = self.stamp
                    image.header.frame_id = self.frame_id
                except CvBridgeError as e:
                    rospy.logerr(e)
                    continue

                if self.img_pub.get_num_connections():             
                    self.img_pub.publish(image)
                else:
                    rospy.logwarn("No nodes subscribed to {}".format(self.img_pub.name))
                
                # Prepare ROS message (target pose)
                detected_marker = deepcopy(self.detected_marker)
                if detected_marker is not None:
                    t, q = detected_marker.get_pose()
                    target = PoseStamped()
                    target.header.stamp = self.stamp
                    target.header.frame_id = self.frame_id
                    target.pose.position.x = t[0]
                    target.pose.position.y = t[1]
                    target.pose.position.z = t[2]
                    target.pose.orientation.x = q[0]
                    target.pose.orientation.y = q[1]
                    target.pose.orientation.z = q[2]
                    target.pose.orientation.w = q[3]

                    if self.pose_pub.get_num_connections():
                        self.pose_pub.publish(target)

                # Reset values
                self.image = None
                self.detected_marker = None
                self.stamp = None
                self.frame_id = None

            self.rate.sleep()

if __name__ == "__main__":
    qr_tracker = QRDetectorNode(True)
    try:
        qr_tracker.run()
    except rospy.ROSInterruptionException:
        pass 
