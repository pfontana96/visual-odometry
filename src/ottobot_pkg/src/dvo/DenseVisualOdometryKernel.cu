#include <DenseVisualOdometryKernel.cuh>

/*--------------------------------   KERNELS WRAPPERS   -------------------------------*/
void call_step_kernel(  unsigned char* gray,
                        unsigned char* gray_prev,
                        unsigned short* depth_prev,
                        float* residuals,
                        const float T[4][4],
                        const float cam_mat[3][3],
                        const float scale,
                        const int width,
                        const int height)
{
    // We can use pointers passed by CPU process because we're using Unified Memory     
    // Valid because on Nano devices CPU and GPU share physical memory

    // Calculate Nb of threads per block and blocks per grid
    int gx = std::ceil((float) width/CUDA_BLOCKSIZE), gy = std::ceil((float) height/CUDA_BLOCKSIZE);
    dim3 block(CUDA_BLOCKSIZE, CUDA_BLOCKSIZE), grid(gx, gy);
    printf("(w: %d, h: %d) Grid: (%d, %d) | Block: (%d, %d)", width, height, gx, gy, CUDA_BLOCKSIZE, CUDA_BLOCKSIZE);

    step_kernel<<<grid, block>>>(gray,
                                 gray_prev,
                                 depth_prev,
                                 residuals,
                                 T,
                                 cam_mat,
                                 scale,
                                 width,
                                 height);

    // Check for CUDA errors after launching the kernel
    HANDLE_CUDA_ERROR(cudaGetLastError());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}                        

/*------------------------------------   KERNELS   -----------------------------------*/

__global__ void step_kernel(const unsigned char* gray,
                            const unsigned char* gray_prev,
                            const unsigned short* depth_prev,
                            float* residuals,
                            const float T[4][4],
                            const float cam_mat[3][3],
                            const float scale,
                            const int width,
                            const int height)
{
    int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
    int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
    
    // Check bounds
    if(tidx>=width || tidy>=height)
        return;

    int pixel = (tidy*width) + tidx;
    residuals[pixel] = (float) (gray[pixel]);
    // residuals[pixel] = compute_residual(gray,
    //                                     tidx,
    //                                     tidy,
    //                                     gray_prev[pixel],
    //                                     depth_prev[pixel],
    //                                     T,
    //                                     cam_mat,
    //                                     scale,
    //                                     width,
    //                                     height);
    
}

/*------------------------------------  FUNCTIONS  -----------------------------------*/

/*
 Bilinear interpolation
 Arguments:
 ---------
     x: array of x-coordinates to interpolate
     y: array of y-coordinates to interpolate
     dest: array of interpolation results (same dims as x and y)
     src_img: Original image
     width: src_img width
     height: src_img height
     N: Nb of points (dim of x, y, and dest)
*/
__device__ float bilinear_interpolation(   const float x, 
                                            const float y, 
                                            const unsigned char* src_img,
                                            const int width,
                                            const int height)
{
    int x0 = (int) floorf(x), y0 = (int) floorf(y);
    int x1 = x0 + 1, y1 = y0 + 1;   
    
    // Clip coordinates
    x0 = max(0, min(x0, width-1));
    y0 = max(0, min(y0, height-1));
    x1 = max(0, min(x1, width-1));
    y1 = max(0, min(y1, height-1));

    int Ia = (int) src_img[y0*width + x0];
    int Ib = (int) src_img[y1*width + x0];
    int Ic = (int) src_img[y0*width + x1];
    int Id = (int) src_img[y1*width + x1];
    
    // Calculate weights
    float wa = (x1 - x)*(y1 - y);
    float wb = (x1 - x)*(y - y0);
    float wc = (x - x0)*(y1 - y);
    float wd = (x - x0)*(y - y0);

    float result = (float) (wa*Ia + wb*Ib + wc*Ic + wd*Id);

    return result;
}                                            

__device__ float compute_residual(const unsigned char* gray,
                                  const int tidx,
                                  const int tidy,
                                  const unsigned char gray_prev,
                                  const unsigned short depth_prev,
                                  const float T[4][4],
                                  const float cam_mat[3][3],
                                  const float scale,
                                  const int width,
                                  const int height)
{   
    // Camera matrix
    // [fx  0 cx]
    // [ 0 fy cy]
    // [ 0  0  1]
    float fx = cam_mat[0][0], fy = cam_mat[1][1], cx = cam_mat[0][2], cy = cam_mat[1][2];

    float z = (float) depth_prev; 
    if(z >= 0.0)
    {   
        // Deproject prev image into 3d space
        float x = z*(tidx - cx)/fx, y = z*(tidy - cy)/fy;

        // Transform point
        float x_new, y_new, z_new;
        x_new = T[0][0]*x + T[0][1]*y + T[0][2]*z + T[0][3];
        y_new = T[1][0]*x + T[1][1]*y + T[1][2]*z + T[1][3];
        z_new = T[2][0]*x + T[2][1]*y + T[2][2]*z + T[2][3];

        // Project transformed point into new image
        float x_warped = (fx*x_new/z_new) + cx, y_warped = (fy*y_new/z_new);

        if(!( isnan(x_warped) || isnan(y_warped) ))
        {
            // Compute residuals
            float gray_projected = bilinear_interpolation(x_warped, y_warped, gray, width, height);
            return gray_projected - (float) gray_prev;
        }
    }

    return 0.0;
}                                  

