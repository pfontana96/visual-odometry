#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import logging
import time

import pyrealsense2 as rs
import cv2
import numpy as np
import yaml

from realsense_d435i import RealsenseD435i


logger = logging.getLogger(__name__)


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("data_dir", type=str, help="Dir where images will be stored")

    args = parser.parse_args()

    p = Path(args.data_dir).resolve()

    return p


def init_camera():
    enable_rgb = True
    enable_depth = True
    enable_imu = False
    create_pc = False
    fps = 15

    ctx = rs.context()
    devices = ctx.query_devices()
    cameras = []
    for dev in devices:
        dev_id = dev.get_info(rs.camera_info.serial_number)
        logger.info("Device found '{}' ({})".format(dev.get_info(rs.camera_info.name), dev_id))
        logger.info("Resetting..")
        dev.hardware_reset()
        time.sleep(2)
        logger.info("Done")
        cameras.append(RealsenseD435i(ctx, fps, 
                                      enable_rgb, 
                                      enable_depth,
                                      enable_imu,
                                      create_pc,
                                      dev_id))

    return cameras[0]


def main():
    output_dir = parse_arguments()
    output_dir.mkdir(exist_ok=True, parents=True)

    camera = init_camera()

    # Dump intrinsics
    camera_matrix, coeffs, depth_scale = camera.get_intrinsics()

    intrinsics_file = output_dir / "camera_intrinsics.yaml"
    data = {
        "intrinsics": camera_matrix.tolist(),
        "depth_scale": depth_scale,
        "coeffs": coeffs.tolist()
    }
    with intrinsics_file.open("w") as fp:
        yaml.dump(data, fp)

    # Read camera
    color_image, depth_image, points, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = camera.poll()

    logger.info("Please enter image name")
    image_name = input()

    color_image_path = (output_dir / "{}_rgb.png".format(image_name))
    depth_image_path = (output_dir / "{}_depth.png".format(image_name))

    cv2.imwrite(str(color_image_path), color_image)
    cv2.imwrite(str(depth_image_path), depth_image)
    


if __name__ == "__main__":
    main()