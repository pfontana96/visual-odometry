# This auxiliary file contains a numpy implementation of some functions for manipulating
# Special Orthogonal and Special Eucledian groups (Lie's algebra)
# It was based on Tim Barfoot's book available on:
# http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf

import numpy as np

__lie_epsilon = 1e-6

def limit_angle(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi

# Special Orthogonal Groups

def so3_hat(phi):
    assert type(phi) == np.ndarray, "'phi' should be a numpy array"
    assert phi.shape == (3,1), "Expected shape of (3,1) for 'phi', got {} instead".format(phi.shape)
    hat = np.zeros((3,3), dtype=np.float32)
    hat[0,1] = -phi[2]
    hat[0,2] = phi[1]
    hat[1,0] = phi[2]
    hat[1,2] = -phi[0]
    hat[2,0] = -phi[1]
    hat[2,1] = phi[0]

    return hat

def so3_exp(phi):
    """
    Exponential mapping from so(3) to SO(3), i.e, R3 --> R3x3
    """
    assert type(phi) == np.ndarray, "'phi' should be a numpy array"
    assert phi.shape == (3,1), "Expected shape of (3,1) for 'phi', got {} instead".format(phi.shape)
    theta = np.linalg.norm(phi)

    # Check for 0 rad rotation
    if np.abs(theta) < __lie_epsilon:
        return np.eye(3)
    a = phi / theta

    # After normalizing the axis-angle vector, wrap the angle between [-pi;pi]
    theta = limit_angle(theta)

    return np.cos(theta)*np.eye(3, dtype=np.float32) + (1 - np.cos(theta))*np.dot(a, a.T) + np.sin(theta)*so3_hat(a)

def so3_ln(C):
    """
    Logarithmic mapping from SO(3) to so(3), i.e, R3x3 --> R3
    """
    assert type(C) == np.ndarray, "'C' should be a numpy array"
    assert C.shape == (3,3), "Expected shape (3,3) for 'C', got {} instead".format(C.shape)
    
    w, v = np.linalg.eig(C)
    ids = np.where(np.abs(w - 1) < __lie_epsilon)[0]

    if not ids.size:
        raise ValueError("Rotation matrix has no eigen value '1' (invalid)")

    theta = np.arccos((np.trace(C) - 1)/2)

    # We wrap the angle between [-pi;pi]
    theta = limit_angle(theta)

    # There might be multiple solutions to the eigen problem, we choose the first one
    return theta*v[:,ids[0]]

def so3_ljac(phi):
    """
    Computes SO(3) left jacobian
    """
    assert type(phi) == np.ndarray, "'phi' should be a numpy array"
    assert phi.shape == (3,1), "Expected shape of (3,1) for 'phi', got {} instead".format(phi.shape)
    theta = np.linalg.norm(phi)
    a = phi / theta

    # Check for singularity at theta = 0
    if theta < __lie_epsilon:
        # Use first order Taylor's expansion
        return np.eye(3, dtype=np.float32) + 0.5*so3_hat(phi)

    sin_theta = np.sin(theta)/theta
    return sin_theta*np.eye(3, dtype=np.float32) + (1 - sin_theta)*np.dot(a, a.T) + ((1 - np.cos(theta))/theta)*so3_hat(a)

def so3_inv_ljac(phi):
    assert type(phi) == np.ndarray, "'phi' should be a numpy array"
    assert phi.shape == (3,1), "Expected shape of (3,1) for 'phi', got {} instead".format(phi.shape)
    theta = np.linalg.norm(phi)
    a = phi / theta

    # Check for singularity at theta = 0
    if theta < __lie_epsilon:
        # Use first order Taylor's expansion
        return np.eye(3, dtype=np.float32) + 0.5*so3_hat(phi)

    theta2 = theta/2
    cotan_theta2 = 1/np.tan(theta2)

    return theta2*cotan_theta2*np.eye(3, np.float32) + (1 - theta2*cotan_theta2)*np.dot(a, a.T) - theta2*so3_hat(a)

# Special Eucledian Groups

def se3_hat(xi):
    assert type(xi) == np.ndarray, "'xi' should be a numpy array"
    assert xi.shape == (6,1), "Expected shape of (6,1) for 'xi', got {} instead".format(xi.shape)

    hat = np.zeros((4,4), dtype=np.float32)
    hat[:3,:3] = so3_hat(xi[3:,:])
    hat[:3,3] = xi[:3,0]

    return hat
    
def se3_chat(xi):
    assert type(xi) == np.ndarray, "'xi' should be a numpy array"
    assert xi.shape == (6,1), "Expected shape of (6,1) for 'xi', got {} instead".format(xi.shape)

    phi_hat = so3_hat(xi[3:,:])
    p_hat = so3_hat(xi[:3,:])

    chat = np.zeros((6,6), dtype=np.float32)
    chat[:3,:3] = phi_hat
    chat[3:,3:] = phi_hat
    chat[:3,3:] = p_hat

    return chat

def se3_exp(xi):
    """
    Exponential mapping from se(3) to SE(3), i.e, R6 --> R4x4
    """
    assert type(xi) == np.ndarray, "'xi' should be a numpy array"
    assert xi.shape == (6,1), "Expected shape of (6,1) for 'xi', got {} instead".format(xi.shape)

    xi_hat = se3_hat(xi)

    theta = np.linalg.norm(xi[3:,0])

    # Check for singularity at 0 deg rotation
    if theta < __lie_epsilon:
        T = np.eye(4, dtype=np.float32)
        T[:3,3] = xi[:3,0]

    else:
        xi_hat_2 = np.dot(xi_hat, xi_hat)
        T = np.eye(4, dtype=np.float32) + xi_hat + ((1 - np.cos(theta))/(theta**2))*xi_hat_2 + ((theta - np.sin(theta))/(theta**3))*np.dot(xi_hat_2, xi_hat)

    return T

def se3_ln(T):
    """
    Logarithmic mapping from SE(3) to se(3), i.e, R4x4 --> R6
    """
    assert type(T) == np.ndarray, "'T' should be a numpy array"
    assert T.shape == (4,4), "Expected shape (4,4) for 'T', got {} instead".format(T.shape)

    xi = np.zeros((6,1), dtype=np.float32)
    xi[3:,0] = so3_ln(T[:3,:3])

    # Check for 0 rad rotation
    if np.abs(np.linalg.norm(xi[3:,0])) < __lie_epsilon:
        xi[:3,0] = T[:3,3]
    else:
        inv_J = so3_inv_ljac(xi[3:,:])
        xi[:3,0] = np.dot(inv_J, T[:3,3])

    return xi