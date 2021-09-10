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

def quaternion_from_aa(rot_vec):
    """
    Converts axis angle representation to quaternion (w in last place)
    quat = [x, y, z, w]
    """
    a2 = np.linalg.norm(rot_vec)/2 # angle/2
    sin_a2 = np.sin(a2) # sin(angle/2)

    q = [0] * 4
    q[0] = rot_vec[0] * sin_a2
    q[1] = rot_vec[1] * sin_a2
    q[2] = rot_vec[2] * sin_a2
    q[3] = np.cos(a2)

    return q