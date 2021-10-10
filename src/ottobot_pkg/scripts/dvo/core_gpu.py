import numpy as np 
from scipy.linalg import cho_solve
import cv2
from copy import deepcopy
from pathlib import Path

import pycuda.driver as cuda 
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cm
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule

# import skcuda.linalg as linalg

from .lie_algebra import *

# linalg.init()
__code_file = Path(__file__).resolve().parent.joinpath("src/dvo_cuda.cu")
__inc_file = Path(__file__).resolve().parent.joinpath("inc/dvo_cuda.cuh")
with open(__code_file) as f:
    __mod = SourceModule(f.read(), no_extern_c=False)

# Thread block size
__BLOCK_SIZE = 32

# User defined kernels

def weighting(residuals: gpuarray.GPUArray):
    """
    t-distribution weighting
    """
    sigma_init = 5 # Taken from paper
    dof_default = 5 # Taken from paper
    tol = 1e-3

    n = residuals.size
    
    # First we resolve for the variance 
    residuals2 = residuals**2
    var_last = sigma_init**2
    var = gpuarray.sum(residuals2*(dof_default + 1)/(dof_default  + residuals2/var_last))[0]
    while abs(var - var_last) > tol:
        var_last = var
        var = gpuarray.sum(residuals2*(dof_default + 1)/(dof_default  + residuals2/var_last))[0]/n

    return (dof_default + 1)/(dof_default  + residuals2/var)       

def bilinear_interpolation(img: np.ndarray, x: np.ndarray, y: np.ndarray):
    """
    Interpolate an intensity value bilinearly (useful if warped point is not an int)
    """
    assert x.size == y.size, "x and y should have the same length"

    height, width = img.shape
    N = x.size
    out = np.empty(N).astype(np.float32)

    interp2d_kernel = __mod.get_function("interp2d_kernel")

    dx, mx = divmod(N, __BLOCK_SIZE)
    gdim = ((dx *__BLOCK_SIZE) + mx, 1)

    interp2d_kernel(cuda.In(x), cuda.In(y), cuda.Out(out), cuda.In(img),
                    np.int32(width), np.int32(height), np.int32(N), 
                    block = (__BLOCK_SIZE, 1, 1), grid = gdim)

    # interp2d_kernel(cuda.In(img), cuda.Out(out), np.int32(width), np.int32(height), 
    #                 block = (__BLOCK_SIZE, 1, 1), grid = gdim)
    
    # print("out: {} img: {}".format(out.shape, img.shape))
    return out

def calculate_grads(img):

    height, width = img.shape

    img = img.astype(np.uint8)
    grad_x = np.zeros_like(img).astype(np.float32)
    grad_y = np.zeros_like(img).astype(np.float32)

    dx, mx = divmod(width, __BLOCK_SIZE)
    dy, my = divmod(height, __BLOCK_SIZE)
    gdim = ((dx*__BLOCK_SIZE) + mx, (dy*__BLOCK_SIZE) + my)

    # Using Sobel's method (3x3 kernel)
    sobel_kernel = __mod.get_function("sobel_kernel")

    sobel_kernel(cuda.In(img), cuda.InOut(grad_x), cuda.InOut(grad_y), np.int32(width), np.int32(height),
                 block = (__BLOCK_SIZE, __BLOCK_SIZE, 1), grid=gdim)
    # grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    # grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    return grad_x, grad_y

def compute_residuals(gray, gray_prev, depth_prev, xi, camera_matrix, scale, keep_dims=False):
        """
        Deprojects last seen images (intensities and depth) into a 3d point cloud, applies
        a transform to such pointcloud (given by xi) and then projects the transformed
        pointcloud into a 2d image to compare intensities with current seen image. Then, it
        computes the error between the projected image and the actual one.

        Arguments:
        ---------
        gray: [np.ndarray] Current intensity image (height, width)
        gray_prev: [np.ndarray] Previous intensity image (height, width)
        depth_prev: [np.ndarray] Previous depth image (height, width)
        xi: [np.ndarray] Camera Pose: (x, y, z, ax, ay, az) where (ax, ay, az) is the axis-angle
            representation of the rotation
        camera_matrix: [np.npdarray] Camera matrix specifying fx, fy, cx, cy
        scale: [float] Depth scale
        keep_dims: [bool] If true, residuals' and J's shape qre (height, width) and (height, width)
                   repectively, (height*width,) and (height*width, 6) otherwise

        Returns:
        -------
        residuals: [np.ndarray] Intenisity residuals (height, width)
        J: [np.ndarray] Jacobian of the current intensity image with respect to xi (height, width, 6)
        """
        height, width = gray.shape

        dx, mx = divmod(width, __BLOCK_SIZE)
        dy, my = divmod(height, __BLOCK_SIZE)
        gdim = ((dx*__BLOCK_SIZE) + mx, (dy*__BLOCK_SIZE) + my)

        residuals = np.empty(height*width).astype(np.float32)
        
        residuals_kernel = __mod.get_function("residuals_kernel")

        T = se3_exp(xi) 

        residuals_kernel(cuda.In(gray), cuda.In(gray_prev), cuda.In(depth_prev),
                         cuda.Out(residuals), cuda.In(T), cuda.In(camera_matrix.astype(np.float32)),
                         np.float32(scale), np.int32(width), np.int32(height), 
                         block = (__BLOCK_SIZE, __BLOCK_SIZE, 1), grid = gdim)

        return residuals
        # cx = camera_matrix[0,2]
        # cy = camera_matrix[1,2]
        # fx = camera_matrix[0,0]
        # fy = camera_matrix[1,1]

        # # Create Pointcloud
        # # point = (x, y, z)

        # height, width = gray_prev.shape 
        
        # z = depth_prev.reshape(-1) * scale
        # x = np.tile(range(width), height).astype(np.float32)
        # y = np.repeat(range(height), width).astype(np.float32)

        # # Get valid pixels mask
        # # mask = np.where(z==0, False, True)
        # mask = z != 0.0

        # y = z[mask] * (y[mask] - cy) / fy
        # x = z[mask] * (x[mask] - cx) / fx
                        
        # pc = np.vstack((x, y, z[mask]))
        
        # # Transform PointCloud
        # T = se3_exp(xi) 
        # pc = np.dot(T[:3,:3], pc) + T[:3,3].reshape(3,1)

        # # Project 3d point cloud down to current camera pose (warped points)
        # x_warped = (fx*pc[0,:]/pc[2,:]) + cx
        # y_warped = (fy*pc[1,:]/pc[2,:]) + cy 

        # # Calculate residuals
        # residuals = np.zeros(height*width, dtype=np.float32)
        # gray_projected = np.full(height*width, np.nan, dtype=np.float32)
        # gray_projected[mask] = bilinear_interpolation(gray, x_warped, y_warped)
        
        # residuals[mask] = gray_prev.reshape(-1)[mask] - gray_projected.reshape(-1)[mask] 

        # # Calculate Jacobian
        # n = len(x)
        # J = np.zeros((height*width, 6), dtype=np.float32)

        # J_pi = np.zeros((n, 2, 6), dtype=np.float32)
        # temp = np.array([[fx/pc[2,:], np.zeros(n), -fx*pc[0,:]/pc[2,:]**2, -fx*pc[0,:]*pc[1,:]/pc[2,:]**2, fx*(1 + pc[0,:]**2/pc[2,:]**2), -fx*pc[1,:]/pc[2,:]],
        #                 [np.zeros(n), fy/pc[2,:], -fy*pc[1,:]/pc[2,:]**2, -fy*(1 + pc[1,:]**2/pc[2,:]**2), fy*pc[0,:]*pc[1,:]/pc[2,:]**2, fy*pc[0,:]/pc[2,:]]])
        # J_pi = np.transpose(temp, [2,0,1])
        
        # gradx, grady = calculate_grads(gray)
        # J_img = np.zeros((n, 2), dtype=np.float32)
        # J_img[:, 0] = gradx.reshape(-1)[mask]
        # J_img[:, 1] = grady.reshape(-1)[mask]

        # # Couldn't find a way to do a numpy implementation of this loop (MemoryError)
        # # because of too large matrices
        # for i in range(n):
        #     if mask[i]:
        #         J[i,:] = np.dot(J_img[i], J_pi[i])

        # if keep_dims:
        #     residuals = residuals.reshape(height, width)
        #     J = J.reshape(height, width, 6)
        
        # return residuals, J

class DenseVisualOdometry(object):

    def __init__(self, camera_matrix, depth_scale, levels=5):
       
        self.camera_matrix = camera_matrix
        self.depth_scale = depth_scale

        self.first_frame = True
        self.gray_prev = None
        self.depth_prev = None

        self.levels = levels # Pyramid levels taken from paper
        self.grays_prev_pyr = [None]*levels
        self.depths_prev_pyr = [None]*levels 
    
    def gauss_newton(self, gray, gray_prev, depth_prev, xi, alpha=1e-6, max_iter=100, tol=1e-8):
        """
        Performs one step of dense visual odometry from last camera pose (tvec and q) to
        the current estimated one (on first frame it returns )
        """

        assert(not self.first_frame), "Cannot do Gauss Newton on first frame"
        
        err_prev = 1e24
        converge = False

        # Perform Gauss-Newton Method for finding next xi
        for _ in range(max_iter):
            residuals, J = compute_residuals(gray, 
                                             gray_prev, 
                                             depth_prev, 
                                             xi,
                                             self.camera_matrix,
                                             self.depth_scale)
            
            weights = weighting(residuals)
            residuals = weights*residuals

            # Compute update step (solving linear system with Cholesky decommposition)
            #  Jt.W.J.delta_xi = -Jt.W.r
            Jt = J.T
            J = weights.reshape(-1,1)*J
            
            b = np.dot(Jt, residuals)
            H = np.dot(Jt, J)
            L = np.linalg.cholesky(H)

            delta_xi = -cho_solve((L, True), b)

            # Update xi
            xi += alpha*delta_xi.reshape(-1,1)

            # Compute error (cuadratic)
            err = np.dot(residuals, residuals.T)
            if np.abs(err - err_prev) < tol:
                converge = True
                break

            err_prev = err
    
        return xi

    def step(self, color, depth, xi):

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        # Create pyramids
        grays_pyr = [None]*self.levels
        grays_pyr[0] = gray 
        depths_pyr = [None]*self.levels
        depths_pyr[0] = depth

        for i in range(1, self.levels):
            grays_pyr[i] = cv2.pyrDown(grays_pyr[i-1])
            depths_pyr[i] = cv2.pyrDown(depths_pyr[i-1])

        if not self.first_frame:
                for lvl in range(self.levels - 1, -1, -1):
                    # Compute xi estimation beginning from least resoluted images
                    # logging.debug("gray[{i}]:{g} | gray_prev[{i}]:{gp} | depth_prev[{i}]:{dp}".format(
                    #              i=lvl, g=self.grays_pyr[lvl].shape, gp=self.grays_prev_pyr[lvl].shape, 
                    #              dp = self.depths_prev_pyr[lvl].shape))
                    
                    # logging.debug("gray[{i}]:{g} | gray_prev[{i}]:{gp} | depth_prev[{i}]:{dp}".format(
                    #              i=lvl, g=(np.min(self.grays_pyr[lvl]), np.max(self.grays_pyr[lvl])), 
                    #              gp=(np.min(self.grays_prev_pyr[lvl]), np.max(self.grays_prev_pyr[lvl])), 
                    #              dp =(np.min(self.depths_prev_pyr[lvl]), np.max(self.depths_prev_pyr[lvl]))))

                    xi = self.gauss_newton(grays_pyr[lvl], self.grays_prev_pyr[lvl],
                                           self.depths_prev_pyr[lvl], xi)

        else:
            self.first_frame = False
        
        self.grays_prev_pyr = grays_pyr
        self.depths_prev_pyr = depths_pyr

        return xi

if __name__ == '__main__':
    from pathlib import Path
    import logging
    import yaml
    from timeit import repeat

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)

    # Load camera intrinsics
    filepath = Path(__file__).resolve().parent.parent.parent.joinpath("config/camera_intrinsics.yaml")
    try:
        with open(filepath, 'r') as f:
            data = yaml.load(f, Loader=yaml.loader.SafeLoader)
        camera_matrix = np.array(data['camera_matrix'])
        coeffs = np.array(data['coeffs'])
    except Exception as e:
        logging.error(e)

    scale = 0.0010000000474974513

    dump_path = Path().home().joinpath("Documents/test_data")

    # # COunt available data
    # ctn = 0
    # for obj in dump_path.iterdir():
    #     if obj.is_file():
    #         ctn += 1

    # n = int(ctn/2)

    # dvo = DenseVisualOdometry(camera_matrix, scale, 2)

    # cv2.namedWindow('Color')
    # cv2.namedWindow('Depth')

    # for i in range(n):
    #     color = cv2.imread(str(dump_path.joinpath("color_{}.png".format(i+1))))
    #     depth = cv2.imread(str(dump_path.joinpath("depth_{}.png".format(i+1))), cv2.IMREAD_ANYDEPTH)
    #     depth_cmap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.09), cv2.COLORMAP_JET)


        # # Step
        # T_init = np.eye(4)
        # xi = se3_ln(T_init)
        # xi = dvo.step(color, depth, xi)
        # logging.info("Xi at {}: {}".format(i+1, xi))

        # # # Displaying
        # # cv2.imshow('Color', color)
        # # cv2.imshow('Depth', depth_cmap)

        # # Press 'q' or ESC to close window
        # key =  cv2.waitKey(1)
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()
        #     break

    color_1 = cv2.imread(str(dump_path.joinpath("color_1.png")), cv2.IMREAD_GRAYSCALE)
    color_2 = cv2.imread(str(dump_path.joinpath("color_2.png")), cv2.IMREAD_GRAYSCALE)
    depth_1 = cv2.imread(str(dump_path.joinpath("depth_1.png")), cv2.IMREAD_ANYDEPTH) 
    depth_2 = cv2.imread(str(dump_path.joinpath("depth_2.png")), cv2.IMREAD_ANYDEPTH)

    # # Display images
    images = np.hstack((color_1, depth_1, color_2, depth_2))
    cv2.imshow('Color images', np.hstack((color_1, color_2)))

    # Test interpolation
    height, width = color_2.shape
    x = np.tile(range(width), height).astype(np.float32)
    y = np.repeat(range(height), width).astype(np.float32)
    # y = np.tile(range(height), width).astype(np.float32)
    # x = np.repeat(range(width), height).astype(np.float32)

    # dummy_img = np.arange(50, dtype=np.uint8).reshape(5,10)
    # img_int = bilinear_interpolation(dummy_img, x, y)
    # print(img_int.astype(np.int8))
    img_int = bilinear_interpolation(color_2.astype(np.uint8), x, y)
    img_int = img_int.reshape(height, width)

    img_int = cv2.convertScaleAbs(img_int, alpha=255/np.max(img_int))
    print("imgint: ({},{})".format(np.min(img_int), np.max(img_int)))
    cv2.imshow("Interpolation", img_int)

    # Test gradients 
    gradx, grady = calculate_grads(color_2)
    gradx = cv2.convertScaleAbs(gradx, alpha=255/np.max(gradx))
    grady = cv2.convertScaleAbs(grady, alpha=255/np.max(grady))
    print("gradx: ({},{})".format(np.min(gradx), np.max(gradx)))
    print("grady: ({},{})".format(np.min(grady), np.max(grady)))
    cv2.imshow("Sobel gradients", np.hstack((gradx, grady)))

    # Test residuals
    xi_init = np.zeros((6,1), dtype=np.float32)
    residuals = compute_residuals(color_2, color_1, depth_1, xi_init, camera_matrix, scale)
    residuals = residuals.reshape(height, width)
    residuals = cv2.convertScaleAbs(residuals, alpha=255/np.max(residuals))

    # Test weights
    weights = weighting(gpuarray.to_gpu(residuals)).get()

    cv2.imshow("Residuals & weights", np.hstock((residuals, weights)))
    #  Press 'q' or ESC to close window
    cv2.waitKey()
    cv2.destroyAllWindows()
