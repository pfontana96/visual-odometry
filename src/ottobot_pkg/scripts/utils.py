import numpy as np

# Can't believe there is not a native ROS way of doing this with Pyhton3 and tf2_ros 
def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = [0] * 4
    q[0] = cy * cp * cr + sy * sp * sr
    q[1] = cy * cp * sr - sy * sp * cr
    q[2] = sy * cp * sr + cy * sp * cr
    q[3] = sy * cp * cr - cy * sp * sr

    return q

def is_rotation_matrix(R: np.ndarray):
    assert R.shape == (3,3), "Not valid Rotation matrix, shape should be (3,3) got {} instead".format(R.shape)
    should_be_identity = np.dot(R.T, R)
    n = np.linalg.norm(np.identity(3, dtype=R.dtype) - should_be_identity)
    return n < 1e-6

def rotmat_to_euler(R):
    assert(is_rotation_matrix(R))
    sy = np.sqrt(R[0,0]**2, R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        ro
    else: