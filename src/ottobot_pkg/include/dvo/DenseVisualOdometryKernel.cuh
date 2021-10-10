#ifndef DVO_CUDA_H
#define DVO_CUDA_H

// Add cudart in case .cpp executable is compiled with gcc
#include <cuda_runtime_api.h>

#include <stdio.h>

static void handle_error(cudaError_t err, const char* file, int line)
{
    if(err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_CUDA_ERROR(err) (handle_error(err, __FILE__, __LINE__))

#define CUDA_BLOCKSIZE 32

/*------------------------------------   KERNELS   -----------------------------------*/

__global__ void step_kernel(const unsigned char* gray,
                            const unsigned char* gray_prev,
                            const unsigned char* depth_prev,
                            float* residuals,
                            const float T[4][4],
                            const float cam_mat[3][3],
                            const float scale,
                            const int width,
                            const int height);

/*------------------------------------  FUNCTIONS  -----------------------------------*/

__device__ float bilinear_interpolation(   const float x, 
                                           const float y, 
                                           const unsigned char* src_img,
                                           const int width,
                                           const int height);

__device__ float compute_residual(const unsigned char* gray,
                                  const int tidx,
                                  const int tidy,
                                  const unsigned char gray_prev,
                                  const unsigned char depth_prev,
                                  const float T[4][4],
                                  const float cam_mat[3][3],
                                  const float scale,
                                  const int width,
                                  const int height);

#endif