#include <stdio.h>
// Declarations

// Structs
struct Pixel {
    float x;
    float y;
};

struct Point3d {
    float x;
    float y;
    float z;
};

// Kernels
__global__ void interp2d_kernel( float* x, 
                                 float* y,
                                 float* dest, 
                                 const unsigned char* src_img,
                                 const int width,
                                 const int height,
                                 const int N);

__global__ void sobel_kernel(const unsigned char* img, 
                             float* gradx, 
                             float* grady, 
                             const int width, 
                             const int height);

__global__ void dummy_kernel(const unsigned char* img_in, int* out, const int width, const int height);

__global__ void residuals_kernel(const unsigned char* gray,
                                 const unsigned char* gray_prev,
                                 const unsigned char* depth_prev,
                                 float* residuals,
                                 const float T[4][4],
                                 const float cam_mat[3][3],
                                 const float scale,
                                 const int width,
                                 const int height);
// Functions
__device__ float interp2d(const float x, 
                          const float y, 
                          const unsigned char* src_img,
                          const int width,
                          const int height);

__device__ Point3d deproject(const Pixel pixel, const float cam_mat[3][3], const float z);

__device__ Pixel project(const Point3d point, const float cam_mat[3][3]);

__device__ Point3d transform(const Point3d point, const float T[4][4]);

// __device__ float compute_residual(const unsigned char* gray,
//                                   const unsigned char* gray_prev,
//                                   const unsigned char* depth_prev,
//                                   const float T[4][4],
//                                   const float cam_mat[3][3],
//                                   const float scale,
//                                   const int width,
//                                   const int height);
// Implementation

__global__ void dummy_kernel(const unsigned char* img_in, int* out, const int width, const int height)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if(tid < (width*height))
    {
        printf("tid: %d | img[tid]: %d\n", tid, (int) img_in[tid]);
        out[tid] = (int) img_in[tid];
    }
}

// Kernels

// Bilinear interpolation
// Arguments:
// ---------
//     x: array of x-coordinates to interpolate
//     y: array of y-coordinates to interpolate
//     dest: array of interpolation results (same dims as x and y)
//     src_img: Original image
//     width: src_img width
//     height: src_img height
//     N: Nb of points (dim of x, y, and dest)

__global__ void interp2d_kernel( float* x, 
                                 float* y,
                                 float* dest, 
                                 const unsigned char* src_img,
                                 const int width,
                                 const int height,
                                 const int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
    {
        dest[tid] = interp2d(x[tid], y[tid], src_img, width, height);
    }
} 

__global__ void sobel_kernel(const unsigned char* img, 
                             float* gradx, 
                             float* grady, 
                             const int width, 
                             const int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if ((x>0) && (y>0) && (x<(width-1)) && (y<(height-1)))
    {
        gradx[y*width + x] = (-1*img[(y-1)*width + (x-1)]) + (-2*img[y*width + (x-1)]) + (-1*img[(y+1)*width + (x-1)]) +
                             (   img[(y-1)*width + (x+1)]) + ( 2*img[y*width + (x+1)]) + (   img[(y+1)*width + (x+1)]);
        grady[y*width + x] = (   img[(y-1)*width + (x-1)]) + ( 2*img[(y-1)*width + x]) + (   img[(y-1)*width + (x+1)]) +
                             (-1*img[(y+1)*width + (x-1)]) + (-2*img[(y+1)*width + x]) + (-1*img[(y+1)*width + (x+1)]);
    }
}

__global__ void residuals_kernel(const unsigned char* gray,
                                 const unsigned char* gray_prev,
                                 const unsigned char* depth_prev,
                                 float* residuals,
                                 const float T[4][4],
                                 const float cam_mat[3][3],
                                 const float scale,
                                 const int width,
                                 const int height)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(tidx<width && tidy<height)
    {
        residuals[tidy*width + tidx]= 0.0;
        Pixel pixel;
        pixel.x = (float) tidx;
        pixel.y = (float) tidy;

        // Deproject image into 3d space if depth is available
        float z = (float) depth_prev[tidy*width + tidx];
        if (z > 0.0)
        {
            Point3d point = deproject(pixel, cam_mat, z);

            // Transform point
            point = transform(point, T);

            // Project transformed point into new image
        }
        // residuals[tidy*width + tidx]= 0.0;
        
        // float fx = cam_mat[0][0], fy = cam_mat[1][1], cx = cam_mat[0][2], cy = cam_mat[1][2];

        // // Deproject prev image into 3d space if depth is available
        // float z = (float) depth_prev[tidy*width + tidx]; 
        // if(z >= 0.0)
        // {
        //     float x = z*(tidx - cx)/fx, y = z*(tidy - cy)/fy;

        //     // Transform point
        //     float x_new, y_new, z_new;
        //     x_new = T[0][0]*x + T[0][1]*y + T[0][2]*z + T[0][3];
        //     y_new = T[1][0]*x + T[1][1]*y + T[1][2]*z + T[1][3];
        //     z_new = T[2][0]*x + T[2][1]*y + T[2][2]*z + T[2][3];

        //     // Project transformed point into new image
        //     float x_warped = (fx*x_new/z_new) + cx, y_warped = (fy*y_new/z_new);

        //     if(!( isnan(x_warped) || isnan(y_warped) ))
        //     {
        //         // printf("thread[%d, %d] warped: (%.2f, %.2f)\n", tidy, tidx, y_warped, x_warped);
        //         // Calculate residuals
        //         float gray_projected = interp2d(x_warped, y_warped, gray, width, height);
        //         residuals[tidy*width + tidx] = gray_projected - (float) gray_prev[tidy*width + tidx];
        //     }
        // }
    }
}

// Functions
__device__ float interp2d(const float x, 
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

    // Image is stored as an array in pitched linear memory in row major order
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

__device__ Point3d deproject(const Pixel pixel, const float cam_mat[3][3], const float z)
{
    // Camera matrix
    // [fx  0 cx]
    // [ 0 fy cy]
    // [ 0  0  1]
    Point3d point;
    float fx = cam_mat[0][0], fy = cam_mat[1][1], cx = cam_mat[0][2], cy = cam_mat[1][2];

    point.x = z*(pixel.x - cx)/fx;
    point.y = z*(pixel.y - cy)/fy;
    point.z = z;

    return point;
}

__device__ Pixel project(const Point3d point, const float cam_mat[3][3])
{
    // Camera matrix
    // [fx  0 cx]
    // [ 0 fy cy]
    // [ 0  0  1]
    Pixel pixel;
    float fx = cam_mat[0][0], fy = cam_mat[1][1], cx = cam_mat[0][2], cy = cam_mat[1][2];

    pixel.x = (fx*point.x)/point.z + cx;
    pixel.y = (fy*point.y/point.z + cy;

    return pixel;
}

__device__ Point3d transform(const Point3d point, const float T[4][4])
{
    Point3d point_new;
    point_new.x = T[0][0]*x + T[0][1]*y + T[0][2]*z + T[0][3];
    point_new.y = T[1][0]*x + T[1][1]*y + T[1][2]*z + T[1][3];
    point_new.z = T[2][0]*x + T[2][1]*y + T[2][2]*z + T[2][3];

    return point_new
}

// __device__ float compute_residual(const unsigned char* gray,
//                                   const unsigned char* gray_prev,
//                                   const unsigned char* depth_prev,
//                                   const float T[4][4],
//                                   const float cam_mat[3][3],
//                                   const float scale,
//                                   const int width,
//                                   const int height)
// {
//     // Camera matrix
//     // [fx  0 cx]
//     // [ 0 fy cy]
//     // [ 0  0  1]
//     float fx = cam_mat[0][0], fy = cam_mat[1][1], cx = cam_mat[0][2], cy = cam_mat[1][2];

//     float z = (float) depth_prev[tidy*width + tidx]; 
//     if(z >= 0.0)
//     {   
//         // Deproject prev image into 3d space
//         float x = z*(tidx - cx)/fx, y = z*(tidy - cy)/fy;

//         // Transform point
//         float x_new, y_new, z_new;
//         x_new = T[0][0]*x + T[0][1]*y + T[0][2]*z + T[0][3];
//         y_new = T[1][0]*x + T[1][1]*y + T[1][2]*z + T[1][3];
//         z_new = T[2][0]*x + T[2][1]*y + T[2][2]*z + T[2][3];

//         // Project transformed point into new image
//         float x_warped = (fx*x_new/z_new) + cx, y_warped = (fy*y_new/z_new);

//         if(!( isnan(x_warped) || isnan(y_warped) ))
//         {
//             // Compute residuals
//             float gray_projected = interp2d(x_warped, y_warped, gray, width, height);
//             return gray_projected - (float) gray_prev[tidy*width + tidx];
//         }
//     }

//     return 0.0;
// }                                  
