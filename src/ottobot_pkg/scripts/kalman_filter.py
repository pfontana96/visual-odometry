import numpy as np

class KalmanFilter6DOF(object):
    """
    for further info visit: http://campar.in.tum.de/Chair/KalmanFilter
    """
    def __init__(self, dt):
        
        self.dt = dt

        # state_vector = [x, y, z, x', y', z', x'', y'', z'', 
        #                 a_x, a_y, a_z, a_x', a_y', a_z', a_x'', a_y'', a_z''] 
        # where (a_x, a_y, a_z) is the axis-angle representation of rotation
        n_states = 18
        n_measurements = 6
        n_inputs = 0


        # Initial state
        self.x = np.zeros((n_states, 1), dtype=float)

        # State transition (dynamic model with acceleration)
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
        self.A = np.eye(n_states, dtype=float)

        # Position
        self.A[0,3] = dt
        self.A[1,4] = dt
        self.A[2,5] = dt
        self.A[3,6] = dt
        self.A[4,7] = dt
        self.A[5,8] = dt
        
        self.A[0,6] = dt**2/2
        self.A[1,7] = dt**2/2
        self.A[2,8] = dt**2/2

        # Orientation
        self.A[9,12]  = dt
        self.A[10,13] = dt
        self.A[11,14] = dt
        self.A[12,15] = dt
        self.A[13,16] = dt
        self.A[14,17] = dt
        
        self.A[9,15]  = dt**2/2
        self.A[10,16] = dt**2/2
        self.A[11,17] = dt**2/2

        # Measurement mapping matrix
        # [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        # [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        # [0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
        # [0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0]
        # [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0]
        # [0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]
        self.H = np.zeros((n_measurements, n_states), dtype=float)
        self.H[0,0] = 1  # x
        self.H[1,1] = 1  # y
        self.H[2,2] = 1  # z
        self.H[3,9]  = 1 # a_x
        self.H[4,10] = 1 # a_y
        self.H[5,11] = 1 # a_z

        # Process Noise Covariance and Measurement Noise Covariance
        # values taken from https://docs.opencv.org/3.3.0/dc/d2c/tutorial_real_time_pose.html
        self.Q = np.eye(n_states, dtype=float)*1e-5
        self.R = np.eye(n_measurements, dtype=float)*1e-4

        # Initial Covariance matrix
        self.P = np.eye(n_states, dtype=float)

    def predict(self):
        # Update time step
        self.x = np.dot(self.A, self.x)

        # Calculate error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q 

        return self.x[:3], self.x[9:12] # translation and rotation

    def update(self, z):
        """
        z: (np.array(float)) array of shape (6,1) containing measured [x, y, z, a_x, a_y, a_z]
        """
        assert(type(z) == np.ndarray), "z should be a numpy array"
        assert(z.shape == (6,1)), "Expected shape (6,1), got {} instead".format(z.shape)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R 

        # Calculate Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = np.dot((I - (K*self.H)), self.P)

        return self.x[:3], self.x[9:12] # translation and rotation
        
class KalmanFilter(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        Numpy implementation of a Kalman filter
        Arguments:
        ---------
        dt: [float] Sampling time in seconds
        u_x: [float] Acceleration on x-axis
        u_y: [float] Acceleration on y-axis
        std_acc: [float] Process noise magnitude
        x_std_meas: [float] Standard deviation of the measurement in x-direction
        y_std_meas: [float] Standard deviation of the measurement in y-direction
        """
        self.dt = dt

        # Control input variables
        self.u = np.array([[u_x], [u_y]])
        
        # Initial state
        # x = [pos_x, pos_y, vel_x, vel_y]
        self.x = np.zeros((4,1))
        
        # State transition
        self.A = np.eye(4)
        self.A[0,2] = self.dt
        self.A[1,3] = self.dt

        # Control input matrix
        self.B = np.array([[self.dt**2/2, 0],
                           [0, self.dt**2/2],
                           [self.dt, 0],
                           [0, self.dt]])

        # Measurement mapping matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Initial process noise covariance
        self.Q = std_acc**2 * np.array([[self.dt**4/4, 0, self.dt**3/2, 0],
                                        [0, self.dt**4/4, 0, self.dt**3/2],
                                        [self.dt**3/2, 0, self.dt**2, 0],
                                        [0, self.dt**3/2, 0, self.dt**2]])

        # Initial measurement noise covariance
        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])

        # Initial covariance matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        # Update time step
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q 

        return self.x[0:2]

    def update(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R 

        # Calculate Kalman gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))
        I = np.eye(self.H.shape[1])

        # Update error covariance matrix
        self.P = (I - (K*self.H)) * self.P

        return self.x[0:2]

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

    n_states = 18 # Nb of states
    n_measurements = 6 # Nb of measurements
    n_inputs = 0 # Nb of action control
    dt = 1/fps

    KF = cv2.KalmanFilter(n_states, n_measurements, n_inputs)

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

    KF.transitionMatrix = t_matrix

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

    KF.measurementMatrix = m_matrix

    # Process Noise
    KF.processNoiseCov = np.eye(n_states, dtype=np.float32)*1e-5

    # Measurement Noise
    KF.measurementNoiseCov = np.eye(n_measurements, dtype=np.float32)*1e-4

    # KF = KalmanFilter6DOF(1/fps)

    # Error Covariance
    # KF.errorCovPost = np.eye()

    # logging.getLogger().setLevel(logging.DEBUG)
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
            # tvec, rvec = KF.predict()
            estimate = KF.predict()
            
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

                            # # Correct Kalman
                            # measurement = np.array(6(6,1), dtype=float)
                            # measurement[:3] = detected_marker.t
                            # measurement[3:] = detected_marker.r 
                            # KF.correct(measurement)   
            # Update Kalman
            # We update the filter with a current valid measurement if there's any
            # the last one otherwise
            if detected_marker is not None:
                measurement = np.empty((6,1), dtype=np.float32)
                measurement[:3] = detected_marker.t.reshape(-1,1)
                measurement[3:] = detected_marker.r.reshape(-1,1)
                # tvec, rvec = KF.update(measurement)
                estimate = KF.correct(measurement)

             # Draw Kalman's estimated axis
            tvec = estimate[:3]
            rvec = estimate[9:12]
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