import ctypes

import numpy as np
import cv2

from ottopy import lib


class RGBDImagePyramid:
    def __init__(self, gray_image: np.ndarray, depth_image: np.ndarray, levels: int):
        lib.RGBDImagePyramid_new.argtypes = [
            np.ctypeslib.ndpointer(ctypes.c_uint8), np.ctypeslib.ndpointer(ctypes.c_uint16), ctypes.c_int
        ]
        self.obj = lib.RGBDImagePyramid_new(gray_image, depth_image, levels)
