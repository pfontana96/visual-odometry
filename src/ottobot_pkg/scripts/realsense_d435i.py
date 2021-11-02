#!/usr/bin/env python3
import pyrealsense2 as rs
import time
import logging
import numpy as np

WIDTH = 424
HEIGHT = 240

class RealsenseD435i(object):
    def __init__(self, context, fps, enable_rgb=True, enable_depth=True, enable_imu=False, create_pc = False, device_id=None):
        
        self.ctx = context

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_imu = enable_imu

        self.device_id = device_id

        # Configure streams

        # IMU
        self.imu_pipeline = None
        if self.enable_imu:
            logging.info("Configuring IMU pipe..")
            self.imu_pipeline = rs.pipeline(self.ctx)
            imu_config = rs.config()
            if self.device_id != None:
                imu_config.enable_device(self.device_id)
            imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
            imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

            # Start IMU stream
            imu_profile = self.imu_pipeline.start(imu_config)
            logging.info("Started pipe, wait for first frames..")

            # Eat some frames to allow IMU to settle
            for i in range(0, 5):
                try:
                    self.imu_pipeline.wait_for_frames(10000)
                except Exception as e:
                    logging.error(e)
                    break
            logging.info("Done")
        # Images streams
        self.pipeline = None
        self.align = None
        self.pointcloud = None

        if self.enable_rgb or self.enable_depth:
            logging.info("Configuring images pipe..")
            self.pipeline = rs.pipeline(self.ctx)
            config = rs.config()
            if self.device_id != None:
                logging.debug("Enabling device..")
                config.enable_device(self.device_id)
            
            if self.enable_depth:
                logging.debug("Enabling depth stream..")
                config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, fps)
                if create_pc:
                    logging.debug("Creating pointcloud object..")
                    self.pointcloud = rs.pointcloud()

            if self.enable_rgb:
                logging.debug("Enabling color stream..")
                config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, fps)

            if self.enable_depth and self.enable_rgb:
                self.align = rs.align(rs.stream.color)

            # Start image streams
            stream_up = False
            attemps = 0
            for i in range(0,5):
                try:
                    self.profile = self.pipeline.start(config)
                    stream_up = True
                except Exception as e:
                    logging.error(e)
                    logging.info("Trying to start pipeline again.. ({}/5)".format(i+1))
                    time.sleep(3)
                if stream_up:
                    break

            if not stream_up:
                raise RuntimeError("Couldn't start pipeline")

            logging.info("Started pipe, wait for first frames..")
            
            # Eat some frames to allow auto-exposure to settle
            for i in range(0, 5):
                try:
                    self.pipeline.wait_for_frames(10000)
                except Exception as e:
                    logging.error(e)
                    break
            logging.info("Done")

        # Let camera warm up
        time.sleep(2)

    def get_intrinsics(self):
        frame = None
        if self.enable_depth:
            frame = self.profile.get_stream(rs.stream.depth)
            depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        elif self.enable_rgb:
            frame = self.profile.get_stream(rs.stream.color)
        
        if frame is None:
            logging.error("No camera stream is enabled")
            return None, None
        
        intrinsics = frame.as_video_stream_profile().intrinsics

        # Create camera matrix
        # [fx 0  cx]
        # [0  fy cy]
        # [0  0  1]
        camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                  [0, intrinsics.fy, intrinsics.ppy],
                                  [0,             0,              1]])
        coeffs = np.array(intrinsics.coeffs)

        return camera_matrix, coeffs, depth_scale

    def shutdown(self):
        if self.imu_pipeline is not None:
            self.imu_pipeline.stop()
        if self.pipeline is not None:
            self.pipeline.stop()

    def poll(self):
        # Frame state
        color_image = None
        depth_image = None
        points = None
        accel_x = None
        accel_y = None
        accel_z = None
        gyro_x = None
        gyro_y = None
        gyro_z = None

        try:
            if self.enable_imu:
                imu_frames = self.imu_pipeline.wait_for_frames(200)
                
            if self.enable_rgb or self.enable_depth:
                frames = self.pipeline.wait_for_frames(200)

        except Exception as e:
            logging.error(e)
            return color_image, depth_frame, points, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

        # Convert camera frames to images
        if self.enable_rgb or self.enable_depth:
            aligned_frames = self.align.process(frames) if self.align is not None else None
            depth_frame = aligned_frames.get_depth_frame() if aligned_frames is not None else frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame() if aligned_frames is not None else frames.get_color_frame()

            # Convert depth to 16bit array, RGB into 8bit planar array
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16) if self.enable_depth else  None
            color_image = np.asanyarray(color_frame.get_data(), dtype=np.uint8) if self.enable_rgb else None
        
            # Create pointcloud
            if self.pointcloud is not None:
                # If rgb image is enabled, create a textured pointcloud
                if self.enable_rgb:
                    self.pointcloud.map_to(color_frame)
                
                # Generate pointcloud and texture mappings
                points = self.pointcloud.calculate(depth_frame)

        # Get IMU values
        if self.enable_imu:
            accel = imu_frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f).as_motion_frame().get_motion_data()
            accel_x = accel.x
            accel_y = accel.y
            accel_z = accel.z

            gyro = imu_frames.first_or_default(rs.stream.gyro, rs.format.motion_xyz32f).as_motion_frame().get_motion_data()
            gyro_x = gyro.x
            gyro_y = gyro.y
            gyro_z = gyro.z

        return color_image, depth_image, points, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z

if __name__ == "__main__":
    import cv2
    enable_rgb = True
    enable_depth = True
    enable_imu = False
    create_pc = True
    fps = 60

    logging.getLogger().setLevel(logging.DEBUG)

    cameras = []
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        dev_id = dev.get_info(rs.camera_info.serial_number)
        logging.info("Device found '{}' ({})".format(dev.get_info(rs.camera_info.name), dev_id))
        logging.info("Resetting..")
        dev.hardware_reset()
        time.sleep(2)
        logging.info("Done")
        cameras.append(RealsenseD435i(ctx, fps, 
                                      enable_rgb, 
                                      enable_depth,
                                      enable_imu,
                                      create_pc,
                                      dev_id))

    if len(cameras) == 0:
        logging.error("No device found...")

    try:
        while True:
            # Read camera
            color_image, depth_image, points, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = cameras[0].poll()
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # print("color image: {} | depth image: {}".format(type(color_image), type(depth_image)))

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) if enable_depth else None

            # Stack images horizontally
            images = None
            if enable_rgb:
                images = np.hstack((color_image, depth_colormap)) if enable_depth else color_image
            elif enable_depth:
                images = depth_image

            if images is not None:
                cv2.imshow('RealSense', images)

            # Getting intrinsincs parameters
            camera_matrix, coeffs = cameras[0].get_intrinsics()
            logging.info("Camera_matrix:\n{}\nCoeffs:\n{}".format(camera_matrix, coeffs))

            # Print IMU
            if enable_imu:
                logging.debug("IMU | accel: ({:2.2f}, {:2.2f}, {:2.2f}) gyro: ({:2.2f}, {:2.2f}, {:2.2f})".format(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z))

            # Plot pointcloud
            if points is not None:
                print(type(points))

            # Press 'q' or ESC to close window
            key =  cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    except Exception as e:
        logging.error(e)
    
    finally:
        cameras[0].shutdown()
