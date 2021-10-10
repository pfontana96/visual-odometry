import numpy as np
from cv2 import KalmanFilter

from utils import quaternion_from_aa

class KalmanFilter6DOF(object):
    """
    for further info visit: http://campar.in.tum.de/Chair/KalmanFilter
    """
    def __init__(self, dt):
        
        """
        dt: [float] Sampling time in seconds
        """
        n_states = 18
        n_measurements = 6
        n_inputs = 0
        self.filter = KalmanFilter(n_states, n_measurements, n_inputs)

        # Dynamic Model
        # [1 0 0 dt  0  0 dt2   0   0 0 0 0  0  0  0   0   0   0]
        # [0 1 0  0 dt  0   0 dt2   0 0 0 0  0  0  0   0   0   0]
        # [0 0 1  0  0  dt   0  0 dt2 0 0 0  0  0  0   0   0   0]
        # [0 0 0  1  0  0  dt   0   0 0 0 0  0  0  0   0   0   0]
        # [0 0 0  0  1  0   0  dt   0 0 0 0  0  0  0   0   0   0]
        # [0 0 0  0  0  1   0   0  dt 0 0 0  0  0  0   0   0   0]
        # [0 0 0  0  0  0   1   0   0 0 0 0  0  0  0   0   0   0]
        # [0 0 0  0  0  0   0   1   0 0 0 0  0  0  0   0   0   0]
        # [0 0 0  0  0  0   0   0   1 0 0 0  0  0  0   0   0   0]
        # [0 0 0  0  0  0   0   0   0 1 0 0 dt  0  0 dt2   0   0]
        # [0 0 0  0  0  0   0   0   0 0 1 0  0 dt  0   0 dt2   0]
        # [0 0 0  0  0  0   0   0   0 0 0 1  0  0 dt   0   0 dt2]
        # [0 0 0  0  0  0   0   0   0 0 0 0  1  0  0  dt   0   0]
        # [0 0 0  0  0  0   0   0   0 0 0 0  0  1  0   0  dt   0]
        # [0 0 0  0  0  0   0   0   0 0 0 0  0  0  1   0   0  dt]
        # [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   1   0   0]
        # [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   1   0]
        # [0 0 0  0  0  0   0   0   0 0 0 0  0  0  0   0   0   1]

        # Transition matrix
        t_matrix = np.eye(n_states, dtype=np.float32)
        
        # Position
        t_matrix[0,3] = dt
        t_matrix[1,4] = dt
        t_matrix[2,5] = dt
        t_matrix[3,6] = dt
        t_matrix[4,7] = dt
        t_matrix[5,8] = dt
        
        t_matrix[0,6] = dt**2/2
        t_matrix[1,7] = dt**2/2
        t_matrix[2,8] = dt**2/2

        # Orientation
        t_matrix[9,12]  = dt
        t_matrix[10,13] = dt
        t_matrix[11,14] = dt
        t_matrix[12,15] = dt
        t_matrix[13,16] = dt
        t_matrix[14,17] = dt
        
        t_matrix[9,15]  = dt**2/2
        t_matrix[10,16] = dt**2/2
        t_matrix[11,17] = dt**2/2

        self.filter.transitionMatrix = t_matrix

        # Measurement model
        # [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        # [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        # [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        # [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
        # [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
        # [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
        m_matrix = np.zeros((n_measurements, n_states), dtype=np.float32)
        m_matrix[0,0] = 1  # x
        m_matrix[1,1] = 1  # y
        m_matrix[2,2] = 1  # z
        m_matrix[3,9]  = 1 # roll
        m_matrix[4,10] = 1 # pitch
        m_matrix[5,11] = 1 # yaw

        self.filter.measurementMatrix = m_matrix

        # Process Noise
        self.filter.processNoiseCov = np.eye(n_states, dtype=np.float32)*1e-5

        # Measurement Noise
        self.filter.measurementNoiseCov = np.eye(n_measurements, dtype=np.float32)*1e-4

    def predict(self, quaternion=True):
        """
        Performs prediction of Kalman filter
        
        Argeuments:
        -----------
        quaternion: [bool] If True then target's frame rotation is presented by its 
                           quaternion, otherwise axis-angle representation is used

        Return:
        ------
        tvec: [np.ndarray] Array containing translation of target's reference frame
                           [x,y,z]
        rvec: [np.ndarray] Array containing the quaternion representation of the 
                           orientation of target's reference frame [x,y,z,w] if 
                           quaternion is True, axis-angle representation otherwise
        """
        assert type(quaternion) == bool, "'quaternion' should be a boolean"
        estimate = self.filter.predict()
        if quaternion is True:
            return (estimate[:3], np.array(quaternion_from_aa(estimate[9:12]), dtype=np.float32))
        else:
            return (estimate[:3], estimate[9:12])
    
    def correct(self, z, quaternion=True):
        """
        Performs update step of Kalman Filter.

        Arguments:
        ----------
        z: [np.ndarray] Array (6,1) containing [x, y, z, a_x, a_y, a_z] where (a_x, a_y, a_z)
                        is the axis-angle representation of the measured reference frame
        quaternion: [bool] If True then target's frame rotation is presented by its 
                           quaternion, otherwise axis-angle representation is used

        Return:
        ------
        tvec: [np.ndarray] Array containing translation of target's reference frame
                           [x,y,z]
        rvec: [np.ndarray] Array containing the quaternion representation of the 
                           orientation of target's reference frame [x,y,z,w] if 
                           quaternion is True, axis-angle representation otherwise
        """
        assert type(quaternion) == bool, "'quaternion' should be a boolean"
        assert type(z) == np.ndarray, "'z' should be a np.ndarray, got {} instead".format(type(z))
        assert (z.shape == (6,1)) or (z.shape == (1,6)), "'z' shape should be (6,1) or (1,6), got {} instead".format(z.shape)
        estimate = self.filter.correct(z)
        if quaternion is True:
            return (estimate[:3], np.array(quaternion_from_aa(estimate[9:12]), dtype=np.float32))
        else:
            return (estimate[:3], estimate[9:12])

if __name__ == "__main__":
    import cv2
    import pyrealsense2 as rs
    import logging
    import time
    from realsense_d435i import RealsenseD435i
    from qr_detector_node import ArucoMarker
    from pathlib import Path
    import yaml

    enable_rgb = True
    enable_depth = True
    enable_imu = False
    create_pc = False
    fps = 15

    # Create Kalman Filter
    KF = KalmanFilter6DOF(1/fps)

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    cameras = []
    ctx = rs.context()
    devices = ctx.query_devices()
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

    if len(cameras) == 0:
        logger.error("No device found...")

    cv2.namedWindow('KalmanFilter')

    # Aruco params
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    ARUCO_TARGET_ID = 2 # Target ArUco ID

    # Load camera intrinsics
    calib_filename = Path(__file__).resolve().parent.parent.joinpath("config/camera_intrinsics.yaml")
    camera_matrix = None
    coeffs = None
    detected_marker = None
    try:
        with open(calib_filename, 'r') as f:
            data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        camera_matrix = np.array(data['camera_matrix'])
        coeffs = np.array(data['coeffs'])
    except Exception as e:
        logging.error(e)

    try:
        while True:
            # Read camera
            image, depth_image, points, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = cameras[0].poll()
            # Using ArUco
            corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, 
                                                            parameters=aruco_params)
            
            # Kalman Filter steps
            tvec, rvec = KF.predict(False)
            
            # At least one code detected
            if len(corners) > 0:
                ids = ids.flatten()
                
                # Loop over detected markers
                for (m_corners, m_id) in zip(corners, ids):
                    
                    # If detected marker is target
                    if m_id == ARUCO_TARGET_ID:

                        if (camera_matrix is not None) and (coeffs is not None):
                            # Estimate marker pose
                            rot_vec, trans_vec = cv2.aruco.estimatePoseSingleMarkers(m_corners, 0.165, camera_matrix, coeffs)
                            detected_marker = ArucoMarker(ARUCO_TARGET_ID, 
                                                          m_corners.reshape((4,2)).astype(int), 
                                                          trans_vec.reshape(-1), 
                                                          rot_vec.reshape(-1))

                            # Draw detected contours only
                            image = detected_marker.draw_marker(image)

            # Update Kalman
            # We update the filter with a current valid measurement if there's any
            # the last one otherwise
            if detected_marker is not None:
                measurement = np.empty((6,1), dtype=np.float32)
                measurement[:3] = detected_marker.t.reshape(-1,1)
                measurement[3:] = detected_marker.r.reshape(-1,1)
                tvec, rvec = KF.correct(measurement, False)

             # Draw Kalman's estimated axis
            image = cv2.aruco.drawAxis(image, 
                                       camera_matrix,
                                       coeffs,
                                       rvec, # Translation
                                       tvec, # Orientation
                                       0.1)

            cv2.imshow('KalmanFilter', image)

            # Press 'q' or ESC to close window
            key =  cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    except Exception as e:
        logger.error(e)
    
    finally:
        cameras[0].shutdown()