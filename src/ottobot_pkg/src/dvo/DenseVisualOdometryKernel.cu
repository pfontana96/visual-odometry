#include <DenseVisualOdometryKernel.cuh>

__constant__ float CM[3][3]; // Camera matrix is not expected to change over time

/*--------------------------------   KERNELS WRAPPERS   -------------------------------*/
void call_residuals_kernel(     unsigned char* gray,
                                unsigned char* gray_prev,
                                unsigned short* depth_prev,
                                float* residuals,
                                const float* T,
                                float scale,
                                int width,
                                int height)
{
    // We can use pointers passed by CPU process because we're using Unified Memory     
    // Valid because on Nano devices CPU and GPU share physical memory
    dim3 block(CUDA_BLOCKSIZE, CUDA_BLOCKSIZE), grid;
    grid.x = (width + block.x - 1)/block.x;
    grid.y = (height + block.y - 1)/block.y;

    residuals_kernel<<<grid, block>>>(gray,
                                 gray_prev,
                                 depth_prev,
                                 residuals,
                                 T,
                                 scale,
                                 width,
                                 height);

    // Check for CUDA errors after launching the kernel
    HANDLE_CUDA_ERROR(cudaGetLastError());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

void call_weighting_kernel(const float* residuals, 
                            float* weights,
                            float* var,
                            float dof ,
                            int width, 
                            int height)
{
    // We can use pointers passed by CPU process because we're using Unified Memory     
    // Valid because on Nano devices CPU and GPU share physical memory
    dim3 block(CUDA_BLOCKSIZE, CUDA_BLOCKSIZE), grid;
    grid.x = (width + block.x - 1)/block.x;
    grid.y = (height + block.y - 1)/block.y;
    int i=0;

    float var_prev;

    do {
        i++;
        var_prev = *var;
        *var = 0;
        variance_kernel<<<grid, block>>>( residuals,
                                          var_prev,
                                          var,
                                          dof,
                                          width,
                                          height);

        // Check for CUDA errors after launching the kernel
        HANDLE_CUDA_ERROR(cudaGetLastError());
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    } while((*var) - var_prev > 1e-3);

    // Update Weights with found variance
    weighting_kernel<<<grid, block>>>(  residuals,
                                        weights,
                                        (*var),
                                        dof,
                                        width,
                                        height);

    // Check for CUDA errors after launching the kernel
    HANDLE_CUDA_ERROR(cudaGetLastError());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

void call_jacobian_kernel(  const unsigned char* gray,
                            const unsigned short* depth_prev,
                            float* J,
                            const float* T,
                            float scale,
                            int width,
                            int height)
{
    // We can use pointers passed by CPU process because we're using Unified Memory     
    // Valid because on Nano devices CPU and GPU share physical memory
    dim3 block(CUDA_BLOCKSIZE, CUDA_BLOCKSIZE), grid;
    grid.x = (width + block.x - 1)/block.x;
    grid.y = (height + block.y - 1)/block.y;

    // Compute Gradients
    int size = width*height*sizeof(float);
    float *gradx, *grady;
    HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &gradx, size));
    HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &grady, size));
    gradients_kernel<<<grid, block>>>(gray,
                                      gradx,
                                      grady,
                                      width,
                                      height);

    // Check for CUDA errors after launching the kernel
    HANDLE_CUDA_ERROR(cudaGetLastError());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    // Compute Jacobian
    jacobian_kernel<<<grid, block>>> (gray,
                                     depth_prev,
                                     gradx,
                                     grady,
                                     J,
                                     T,
                                     scale,
                                     width,
                                     height);
    
    // Check for CUDA errors after launching the kernel
    HANDLE_CUDA_ERROR(cudaGetLastError());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    HANDLE_CUDA_ERROR(cudaFree(gradx));
    HANDLE_CUDA_ERROR(cudaFree(grady));

}

void call_create_H_matrix_kernel(   const float* J,
                                    float* H,
                                    int size)
{
    dim3 block(CUDA_BLOCKSIZE, 1), grid;
    grid.x = (size + block.x - 1)/block.x;
    grid.y = 1;

    create_H_matrix_kernel<<<grid, block>>> (J, H, size);

    // Check for CUDA errors after launching the kernel
    HANDLE_CUDA_ERROR(cudaGetLastError());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}                                    

float call_newton_gauss_kernel( const float* residuals,
                                const float* weights,
                                const float* J,
                                float* H,
                                float* b,
                                int size)
{
    float* error_ptr;
    HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &error_ptr, sizeof(float)));

    dim3 block(CUDA_BLOCKSIZE, 1), grid;
    grid.x = (size + block.x - 1)/block.x;
    grid.y = 1;

    newton_gauss_kernel<<<grid, block>>>(residuals,
                                         weights,
                                         J,
                                         H,
                                         b,
                                         error_ptr,
                                         size);

    // Check for CUDA errors after launching the kernel
    HANDLE_CUDA_ERROR(cudaGetLastError());
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    float error = *error_ptr;
    HANDLE_CUDA_ERROR(cudaFree(error_ptr));

    return error;
}                                

/*------------------------------------   KERNELS   -----------------------------------*/

__global__ void residuals_kernel(   const unsigned char* gray,
                                    const unsigned char* gray_prev,
                                    const unsigned short* depth_prev,
                                    float* residuals,
                                    const float* T,
                                    float scale,
                                    int width,
                                    int height)
{
    const int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
    const int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
    
    // Check bounds
    if(tidx>=width || tidy>=height)
        return;
    
    float residual = 0.0;
    const int pixel = (tidy*width) + tidx;

    float fx = CM[0][0], fy = CM[1][1], cx = CM[0][2], cy = CM[1][2];
    float z = (float) depth_prev[pixel] * scale;
    if(z>=0)
    {
        // Deproject point into 3d space
        float x = z*(tidx - cx)/fx, y = z*(tidy - cy)/fy;

        // Transform point
        float x_new, y_new, z_new;
        x_new = T[0]*x + T[1]*y + T[2]*z + T[3];
        y_new = T[4]*x + T[5]*y + T[6]*z + T[7];
        z_new = T[8]*x + T[9]*y + T[10]*z + T[11];

        // Project transformed point into new image
        float x_warped = (fx*x_new/z_new) + cx, y_warped = (fy*y_new/z_new);

        if(!( isnan(x_warped) || isnan(y_warped) ))
        {
            // Compute residuals
            float gray_projected = bilinear_interpolation(x_warped, y_warped, gray, width, height);
            residual = gray_projected - (float) gray_prev[pixel];
        }
    } 
    residuals[pixel] = residual;
}

__global__ void weighting_kernel(const float* residuals, 
                                 float* weights,
                                 float var,
                                 float dof ,
                                 int width, 
                                 int height)
{
    const int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
    const int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
    
    // Check bounds
    if(tidx>=width || tidy>=height)
        return;
    
    const int pixel = (tidy*width) + tidx;
    weights[pixel] = (dof + 1.0f)/(dof + (residuals[pixel]*residuals[pixel])/var); 
}

__global__ void variance_kernel(const float* residuals,
                                float var_in,
                                float* var,
                                float dof,
                                int width,
                                int height)
{
    const int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
    const int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
    
    // Check bounds
    if(tidx>=width || tidy>=height)
        return;

    const int pixel = (tidy*width) + tidx, N = width*height;
    float delta_var, res_i_squared = residuals[pixel]*residuals[pixel];

    float weight = (dof + 1.0f)/(dof + (res_i_squared)/var_in);

    // Update Variance
    delta_var = res_i_squared*weight/N;

    atomicAdd(var, delta_var);

}                    

__global__ void jacobian_kernel( const unsigned char* gray,
                                 const unsigned short* depth_prev,
                                 const float* gradx,
                                 const float* grady,
                                 float* J,
                                 const float* T,
                                 float scale,
                                 int width,
                                 int height)
{
    const int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
    const int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
    
    // Check bounds
    if(tidx>=width || tidy>=height)
        return;

    const int pixel = (tidy*width) + tidx;

    float fx = CM[0][0], fy = CM[1][1], cx = CM[0][2], cy = CM[1][2];
    float z = (float) depth_prev[pixel] * scale;
    
    // Deproject point into 3d space
    float x = z*(tidx - cx)/fx, y = z*(tidy - cy)/fy;

    // Transform point
    float x_new, y_new, z_new;
    x_new = T[0]*x + T[1]*y + T[2]*z + T[3];
    y_new = T[4]*x + T[5]*y + T[6]*z + T[7];
    z_new = T[8]*x + T[9]*y + T[10]*z + T[11];

    // Project transformed point into new image
    float x_warped = (fx*x_new/z_new) + cx, y_warped = (fy*y_new/z_new);

    float J_row[6] = {0, 0, 0, 0, 0, 0}, J_i;
    if(!( isnan(x_warped) || isnan(y_warped) ))
    {
        float Jw[2][6] = {fx/z_new, 0, -fx*x_new/(z_new*z_new), -fx*(x_new*y_new)/(z_new*z_new), fx*(1 + (x_new*x_new)/(z_new*z_new)), -fx*y_new/z_new,
                          0, fy/z_new, -fy*y_new/(z_new*z_new), -fy*(1 + (y_new*y_new)/(z_new*z_new)), fy*(x_new*y_new)/(z_new*z_new), fy*x_new/z_new};
        float J1[2];
        J1[0] = bilinear_interpolation(x_warped, y_warped, gradx, width, height);
        J1[1] = bilinear_interpolation(x_warped, y_warped, grady, width, height);

        for(int i = 0; i < 6; i++)
        {
            J_i = J1[0]*Jw[0][i] + J1[1]*Jw[1][i];
            if(!isnan(J_i))
                J_row[i] = J_i;
        }
    }

    const int row = pixel*6;
    for(int i = 0; i < 6; i++)
        J[row+i] = J_row[i];
}

__global__ void gradients_kernel(const unsigned char* gray,
                                 float* gradx,
                                 float* grady,
                                 int width,
                                 int height)
{
    const int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
    const int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
    
    // Check bounds
    if(tidx>=width || tidy>=height)
        return;

    const int pixel = (tidy*width) + tidx;

    // Compute gradient on X direction
    gradx[pixel] = 0.5f*(gray[(tidy*width) + (tidx+1)] - gray[(tidy*width) + (tidx-1)]);

    // Compute gradient on Y direction
    grady[pixel] = 0.5f*(gray[((tidy+1)*width) + tidx] - gray[((tidy-1)*width) + tidx]);
}           

// __global__ void create_H_matrix_kernel(const float* J,
//                                        float* H,
//                                        int size)
// {
//     const int tidx = threadIdx.x + (blockIdx.x * blockDim.x);
//     const int tidy = threadIdx.y + (blockIdx.y * blockDim.y);
    
//     // Check bounds
//     if(tidx>=6 || tidy>=6)
//         return;

//     float result = 0.0f;
//     for(int i = 0; i < size; i++)
//     {
//         result += J[i*6 + tidy] * J[i*6 + tidx];
//     }

//     H[tidy*6 + tidx] = result;
// }

__global__ void create_H_matrix_kernel(const float* J,
                                       float* H,
                                       int size)
{
    const int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    // Check Bounds
    if(tid >= size)
        return;

    float H_i;
    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            H_i = J[tid*6+i]*J[tid*6+j];
            atomicAdd(&H[i*6+j], H_i);
        }
    }
}

/*
Prepares matrix H (6x6) and vector b (6x1) for solving system:
    Jt*W*J*delta_xi = -Jt*W*residuals
where H = Jt*W*J and b=-Jt*Jt*residuals
*/
__global__ void newton_gauss_kernel(const float* residuals,
                                    const float* weights,
                                    const float* J,
                                    float* H,
                                    float* b,
                                    float* error,
                                    int size)
{
    const int tid = threadIdx.x + (blockIdx.x * blockDim.x);

    // Check Bounds
    if(tid >= size)
        return;

    float H_i, b_i, error_i;
    for(int i = 0; i < 6; i++)
    {
        b_i = -J[tid*6+i]*weights[tid]*residuals[tid];
        atomicAdd(&b[i], b_i);

        for(int j = 0; j < 6; j++)
        {
            H_i = J[tid*6+i]*weights[tid]*J[tid*6+j];
            atomicAdd(&H[i*6+j], H_i);
        }
    }

    error_i = weights[tid]*residuals[tid]*residuals[tid];
    atomicAdd(error, error_i);
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

__device__ float bilinear_interpolation( const float x, 
                                         const float y, 
                                         const float* src_img,
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
