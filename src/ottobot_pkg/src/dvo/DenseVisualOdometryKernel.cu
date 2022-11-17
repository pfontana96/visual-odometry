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
        // float Jw[2][6] = {fx/z_new, 0, -fx*x_new/(z_new*z_new), -fx*(x_new*y_new)/(z_new*z_new), fx*(1 + (x_new*x_new)/(z_new*z_new)), -fx*y_new/z_new,
        //                   0, fy/z_new, -fy*y_new/(z_new*z_new), -fy*(1 + (y_new*y_new)/(z_new*z_new)), fy*(x_new*y_new)/(z_new*z_new), fy*x_new/z_new};
        
        // Image Jacobian: 
        //      https://robotacademy.net.au/lesson/the-image-jacobian/
        //      https://journals.sagepub.com/doi/10.5772/51833?icid=int.sj-full.text.citing-articles.2&
        float Jw[2][6] = {-fx/z_new, 0, x_warped/z_new, x_warped*y_warped/fy, -(fx + x_warped*x_warped/fx), y_warped,
                          0, -fy/z_new, y_warped/z_new, (fy + y_warped*y_warped/fy), -x_warped*y_warped/fx, -x_warped};
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
    float gradx_i = 0.0f, grady_i = 0.0f;

    // Compute gradient on X direction
    if(tidx>0 && tidx<(width-1))
        gradx_i = 0.5f*(gray[(tidy*width) + (tidx+1)] - gray[(tidy*width) + (tidx-1)]);

    // Compute gradient on Y direction
    if(tidy>0 && tidy<(height-1))
        grady_i = 0.5f*(gray[((tidy+1)*width) + tidx] - gray[((tidy-1)*width) + tidx]);

    gradx[pixel] = gradx_i;
    grady[pixel] = grady_i;
}           

/**
 * Prepares matrix H (6x6) and vector b (6x1) for solving system:
 *   Jt*W*J*delta_xi = -Jt*W*residuals 
 * @param[in] residuals computed residuals (Nx1)
 * @param[in] weights computed weights for each residual (Nx1)
 * @param[in] J computed residuals Jacobian with respecto to camera pose (Nx6)
 * @param[out] H computed (6x6) matrix, H = Jt*W*J where W is an (NxN) matrix with each
 *               element of weights in its diagonal
 * @param[out] b computed (6x1) vector, b=-Jt*Jt*residuals
 * @param[out] error pointer to error value
 * @param[in] size total number of pixels N
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

/**
 * Interpolates a given (x,y) coordinates within an image
 *
 * @param x x-coordinate to interpolate
 * @param y y-coordinate to interpolate
 * @param src_img original image
 * @param width image's width
 * @param height image's height
 * @returns interpolated value
 */
__device__ float bilinear_interpolation( const float x, 
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

/**
 * Interpolates a given (x,y) coordinates within an image
 *
 * @param x x-coordinate to interpolate
 * @param y y-coordinate to interpolate
 * @param src_img original image
 * @param width image's width
 * @param height image's height
 * @returns interpolated value
 */
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
