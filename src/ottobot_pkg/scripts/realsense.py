#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Int8
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import pyrealsense2 as rs
import numpy as np
import cv2

def config_camera(fps):
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
    
    return pipeline, config

class RealSenseCameraNode(object):
    def __init__(self):
        self.name = rospy.get_param("name", "realsense_camera")
        rospy.init_node(self.name)

        self.frame_id = rospy.get_param("frame_id", "/map")
        self.frame_rate = rospy.get_param("frame_rate", 15) # Defaults to 15 fps
        
        self.rgb_pub = rospy.Publisher(self.name+"/rgb_image", 
                                        Image, 
                                        queue_size=10)
        self.depth_pub = rospy.Publisher(self.name+"/depth_image", 
                                        Image, 
                                        queue_size=10)
        
        self.pipeline, self.config = config_camera(self.frame_rate)
    
    def run(self):

        # Start streaming
        profile = self.pipeline.start(self.config)
        rospy.loginfo("Started streaming..")

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        bridge = CvBridge()

        try:
            while not rospy.is_shutdown():
                # Get frameset of color and depth
                frames = self.pipeline.wait_for_frames()
                # frames.get_depth_frame() is a 640x360 depth image

                # Get ROS time (this is NOT actually capturing time)
                now = rospy.get_rostime()

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    rospy.logwarn("Not valid frames, skipping them..")
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
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

                # Publish images
                self.rgb_pub.publish(color_image_msg)
                self.depth_pub.publish(depth_image_msg)

        finally:
            self.pipeline.stop()

if __name__ == "__main__":
    rs_camera_node = RealSenseCameraNode()
    try:
        rs_camera_node.run()
    except rospy.ROSInterruptException:
        pass
