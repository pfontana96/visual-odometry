import logging
import pyrealsense2 as rs
from pathlib import Path
import yaml
import time
import cv2
import numpy as np

from realsense_d435i import RealsenseD435i
from dvo.core import DenseVisualOdometry, compute_residuals

if __name__ == "__main__":
    enable_rgb = True
    enable_depth = True
    enable_imu = False
    create_pc = False
    fps = 60

    dump_images = True
    dump_path = Path().home().joinpath("Documents")
    offline = False

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)

    if dump_images or not offline:
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

        # Getting intrinsincs parameters
        camera_matrix, coeffs, scale = cameras[0].get_intrinsics()
        print("Depth scale: {}".format(scale))

        # Visual Odometry
        dvo = DenseVisualOdometry(camera_matrix, scale)

        xi = np.zeros((6,1), dtype=np.float32)
        first_frame = True
        i = 1
        try:
            while True:
                # Read camera
                color_image, depth_image, points, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = cameras[0].poll()
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                # print("color image: {} | depth image: {}".format(type(color_image), type(depth_image)))

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) if enable_depth else None
            
                # Print IMU
                if enable_imu:
                    logging.debug("IMU | accel: ({:2.2f}, {:2.2f}, {:2.2f}) gyro: ({:2.2f}, {:2.2f}, {:2.2f})".format(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z))

                # Plot pointcloud
                if points is not None:
                    print(type(points))

                # Visual odometry
                if dump_images:
                    cv2.imwrite(str(dump_path.joinpath("color_{}.png".format(i))), color_image)
                    cv2.imwrite(str(dump_path.joinpath("depth_{}.png".format(i))), depth_image)
                    i += 1

                    images = np.hstack((color_image, depth_colormap))
                    if i == 1:
                        break
                    
                elif first_frame:
                    last_depth = depth_image
                    last_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                    logging.debug("img shape: {} | depth shape: {}".format(last_gray.shape, last_depth.shape))
                    first_frame = False
                else:
                    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                    residuals = compute_residuals(gray, last_gray, last_depth, xi, camera_matrix, 2000)
                    last_depth = depth_image
                    last_gray = gray

                    #  Display images
                    images = np.hstack((color_image, depth_colormap, gray, residuals))
                
                cv2.imshow('RealSense', images)

                # Press 'q' or ESC to close window
                key =  cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break

        except Exception as e:
            logging.error(e)
        
        finally:
            cameras[0].shutdown()

    else:
        # Offline processing
        color_1 = cv2.imread(str(dump_path.joinpath("color_1.png")), cv2.IMREAD_GRAYSCALE) 
        color_2 = cv2.imread(str(dump_path.joinpath("color_2.png")), cv2.IMREAD_GRAYSCALE)

        depth_1 = cv2.imread(str(dump_path.joinpath("depth_1.png")), cv2.IMREAD_ANYDEPTH) 
        depth_2 = cv2.imread(str(dump_path.joinpath("depth_2.png")), cv2.IMREAD_ANYDEPTH)

        logging.debug("color shape: {} | depth shape: {}".format(color_1.shape, depth_1.shape))

        images = np.hstack((color_1, depth_1, color_2, depth_2))
        cv2.imshow('Color images', np.hstack((color_1, color_2)))

        depth_cmap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_1, alpha=0.09), cv2.COLORMAP_JET)
        depth_cmap_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_2, alpha=0.09), cv2.COLORMAP_JET)

        cv2.imshow('Depth images', np.hstack((depth_cmap_1, depth_cmap_2)))

        # cv2.imshow('Test', depth_1)
        cv2.waitKey()
        cv2.destroyAllWindows()
        