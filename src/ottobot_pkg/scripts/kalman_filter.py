import numpy as np

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

    # Create Kalman Filter

    enable_rgb = True
    enable_depth = True
    enable_imu = False
    create_pc = False
    fps = 60

    KF = KalmanFilter(1/fps, 1, 1, 1, 0.1, 0.1)

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
    
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Close', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Read camera
            image, depth_image, points, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = cameras[0].poll()
            
            # Detect AR code
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9,9), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # Morph close
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Find contours and filter for QR code
            cnts = cv2.findContours(close, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            centers = []
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                x,y,w,h = cv2.boundingRect(approx)
                area = cv2.contourArea(c)
                ar = w / float(h)
                if len(approx) == 4 and area > 1000 and (ar > .85 and ar < 1.35):
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    image = cv2.drawContours(image, [box], 0, (36,255,12), 2)
                    centers.append(np.array([[x], [y]]))

            if len(centers) > 0:
                (x, y) = KF.predict()
                # Draw prediction
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                (x_1, y_1) = KF.update(centers[0])
                cv2.rectangle(image, (x_1, y_1), (x_1 + w, y_1 + h), (0, 0, 255), 2)
            
            cv2.imshow("RealSense", image)
            cv2.imshow("Close", close)

            # Press 'q' or ESC to close window
            key =  cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    except Exception as e:
        logging.error(e)
    
    finally:
        cameras[0].shutdown()