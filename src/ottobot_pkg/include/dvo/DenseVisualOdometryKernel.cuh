#ifndef DVO_CUDA_H
#define DVO_CUDA_H

// Add cudart in case .cpp executable is compiled with gcc
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <cmath>

#define HANDLE_CUDA_ERROR(err) {handle_cuda_error(err, __FILE__, __LINE__);}
inline void handle_cuda_error(cudaError_t err, const char* file, int line)
{
    if(err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

inline void query_devices()
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    for(int i = 0; i < dev_count; i++)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop, i);
        printf("Found CUDA Capable device %s (%d.%d)\n", dev_prop.name, 
                                                       dev_prop.major,
                                                       dev_prop.minor);
    }
}


#define CUDA_BLOCKSIZE 32

extern __constant__ float CM[3][3]; // Camera matrix is not expected to change over time

/*--------------------------------   KERNELS WRAPPERS   -------------------------------*/

void call_residuals_kernel(     unsigned char* gray,
                                unsigned char* gray_prev,
                                unsigned short* depth_prev,
                                float* residuals,
                                const float* T,
                                float scale,
                                int width,
                                int height);

void call_weighting_kernel( const float* residuals, 
                            float* weights,
                            float* var,
                            float dof ,
                            int width, 
                            int height);

void call_jacobian_kernel(  const unsigned char* gray,
                            const unsigned short* depth_prev,
                            float* J,
                            const float* T,
                            float scale,
                            int width,
                            int height);

float call_newton_gauss_kernel( const float* residuals,
                                const float* weights,
                                const float* J,
                                float* H,
                                float* b,
                                int size);

/*------------------------------------   KERNELS   -----------------------------------*/

__global__ void residuals_kernel( const unsigned char* gray,
                                  const unsigned char* gray_prev,
                                  const unsigned short* depth_prev,
                                  float* residuals,
                                  const float* T,
                                  float scale,
                                  int width,
                                  int height);

__global__ void weighting_kernel(const float* residuals, 
                                 float* weights,
                                 float var,
                                 float dof ,
                                 int width, 
                                 int height);

__global__ void variance_kernel(const float* residuals,
                                float var_in,
                                float* var_out,
                                float dof,
                                int width,
                                int height);

__global__ void jacobian_kernel( const unsigned char* gray,
                                 const unsigned short* depth_prev,
                                 const float* gradx,
                                 const float* grady,
                                 float* J,
                                 const float* T,
                                 float scale,
                                 int width,
                                 int height);

__global__ void gradients_kernel(const unsigned char* gray,
                                 float* gradx,
                                 float* grady,
                                 int width,
                                 int height);

__global__ void newton_gauss_kernel(const float* residuals,
                                    const float* weights,
                                    const float* J,
                                    float* H,
                                    float* b,
                                    float* error,
                                    int size);                                 

/*------------------------------------  FUNCTIONS  -----------------------------------*/

__device__ float bilinear_interpolation(   const float x, 
                                           const float y, 
                                           const unsigned char* src_img,
                                           const int width,
                                           const int height);

__device__ float bilinear_interpolation( const float x, 
                                         const float y, 
                                         const float* src_img,
                                         const int width,
                                         const int height);                                           

#endif
