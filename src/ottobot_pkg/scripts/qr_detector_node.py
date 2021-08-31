#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Int8
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import quaternion_from_matrix

import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass

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
    rot_mat: (np.ndarray) 3x3 rotation matrix
    """
    id: int
    corners: np.ndarray
    trans_vec: np.ndarray
    rot_mat: np.ndarray

    def __init__(self, id: int, corners: np.ndarray, trans_vec: np.ndarray, rot_mat: np.ndarray):
        assert corners.shape == (4,2), "corners shape should be (4,2), got {} instead".format(corners.shape)
        assert trans_vec.shape == (3,), "trans_vec shape should be (3), got {} instead".format(trans_vec.shape)
        assert rot_mat.shape == (3,3), "rot_mat shape should be (3,3), got {} instead".format(rot_mat.shape)
        
        self.id = id
        self.corners = corners 
        self.t = trans_vec
        self.R = rot_mat

    def get_pose(self):
        return self.t, quaternion_from_matrix(self.R)

    

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
        frame_rate = rospy.get_param("~frame_rate", 30)
        filename = rospy.get_param("~calib_filename", "camera_intrinsics.yaml")

        self.img_sub = rospy.Subscriber(img_in, Image, self.callback)
        self.img_pub = rospy.Publisher(img_out, Image, queue_size=5)
        
        self.bridge = CvBridge()
        self.image = None

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
           image = self.bridge.imgmsg_to_cv2(data, "passthrough").astype(np.uint8)
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        height, width, _ = image.shape
        total_area = height * width

        # Using ArUco
        corners, ids, rejected = cv2.aruco.detectMarkers(image, self.aruco_dict, 
                                                        parameters=self.aruco_params)
        
        # At least one code detected
        if len(corners) > 0:
            ids = ids.flatten()
            
            # Loop over detected markers
            for (m_corners, m_id) in zip(corners, ids):
                rospy.loginfo("Marker detected ({}): {}".format(m_id, m_corners))
                
                # If detected marker is target
                if m_id == ARUCO_TARGET_ID:

                    # Markers are returned as (top_left, top_right, bottom_right, bottom_left)
                    image = cv2.drawContours(image, [m_corners.reshape((4, 2)).astype(int)], 0, (36,255,12), 2)

                    if (self.camera_matrix is not None) and (self.coeffs is not None):
                        # Estimate marker pose
                        rot_vec, trans_vec = cv2.aruco.estimatePoseSingleMarkers(m_corners, 0.165, self.camera_matrix, self.coeffs)
                        # rospy.loginfo("Rotation vec:\n{}\nTranslation vec:\n{}".format(rot_vec, trans_vec))
                        
                        image = cv2.aruco.drawAxis(image, self.camera_matrix, self.coeffs, rot_vec, trans_vec, 0.1)

        # Cnnvert modified image to ROS msg
        try:
            self.image = self.bridge.cv2_to_imgmsg(image.astype(np.int8), encoding="passthrough")
            self.image.header.stamp = data.header.stamp
            self.image.header.frame_id = data.header.frame_id

        except CvBridgeError as e:
            rospy.logerr(e)
            return

    def run(self):
        while not rospy.is_shutdown():
            if self.image is not None:
                if self.img_pub.get_num_connections():             
                    self.img_pub.publish(self.image)
                else:
                    rospy.logwarn("No nodes subscribed to {}".format(self.img_pub.name))
                self.image = None

            self.rate.sleep()

if __name__ == "__main__":
    qr_tracker = QRDetectorNode(True)
    try:
        qr_tracker.run()
    except rospy.ROSInterruptionException:
        pass 
