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

def quaternion_from_aa(rvec):
    """
    Converts axis angle representation to quaternion (w in last place)
    quat = [x, y, z, w]
    """
    a2 = np.linalg.norm(rvec)/2 # angle/2
    sin_a2 = np.sin(a2) # sin(angle/2)

    # Normalize rotation vector
    rvec = rvec/np.linalg.norm(rvec)

    q = [0] * 4
    q[0] = rvec[0] * sin_a2
    q[1] = rvec[1] * sin_a2
    q[2] = rvec[2] * sin_a2
    q[3] = np.cos(a2)

    return q

def aa_from_quaternion(q):
    """
    Converts quaternion to axis-angle representation
    """
    angle = 2*np.arccos(q[3])
    s = np.sqrt(1 - q[3]**2)
    epsilon = 1e-4

    if s < epsilon: 
        # To avoid division by zero, if s is close to 0 then the direction of
        # the axis is not relevant
        x = 1
        y = 0
        z = 0
    else:
        x = q[0]/s
        y = q[1]/s
        z = q[2]/s

    rvec = angle * np.array([x, y, z], dtype=np.float32)

    return rvec
