#include <cuda/residuals_kernel.cuh>
#include <cuda_runtime.h>
#include <math.h>

namespace vo {
    namespace cuda {

        __device__ void set_zero_jacobian_row_(float* jacobian, int row) {
            for (size_t i = 0; i < 6; i++){
                jacobian[row + i] = 0.0f;
            }
        };

        __device__ float compute_gradient_gray(
            const uint8_t* image, const uint32_t u, const uint32_t v, bool x_direction, int height, int width
        ) {
            uint32_t prev_pixel, next_pixel;

            if (x_direction == true) {
                prev_pixel = v * width + max(u - 1, 0);
                next_pixel = v * width + min(u + 1, width - 1);

            } else {
                prev_pixel = max(v - 1, 0) * width + u;
                next_pixel = min(v + 1, height) * width + u;
            }

            return 0.5f * ((float) (image[next_pixel] - image[prev_pixel]));
        };

        __device__ float interpolate2dlinear_gray(
            const float x, const float y, const uint8_t* gray_image, int height, int width
        ){
            // NOTE: If not the nvcc compiler chooses the floor function from the standard C++ lib
            float x0 = floor(x), y0 = floor(y);
            float x1 = x0 + 1, y1 = y0 + 1;

            // Return nan if coordiante lies outside of image grid
            if ((x0 < 0) || (y0 < 0) || (x1 >= width) || (y1 >= height)){
                return vo::cuda::nan;
            }

            float w00 = (x1 - x) * (y1 - y), w01 = (x1 - x) * (y - y0), w10 = (x - x0) * (y1 - y), w11 = (x - x0) * (y - y0);

            uint32_t x0_i = (uint32_t) x0, y0_i = (uint32_t) y0, x1_i = (uint32_t) x1, y1_i = (uint32_t) y1;
            float interpolated_value = (
                (w00 * gray_image[y0_i * width + x0_i] + w01 * gray_image[y1_i * width + x0_i] +
                w10 * gray_image[y0_i * width + x1_i] + w11 * gray_image[y1_i * width + x1_i]) /
                ((x1 - x0) * (y1 - y0))
            );

            return interpolated_value;
        }

        __global__ void residuals_kernel(
            const uint8_t* gray_image, const uint8_t* gray_image_prev, const uint16_t* depth_image_prev,
            const float* transform, float fx, float fy, float cx, float cy, float depth_scale, float* residuals_out,
            float* jacobian, int height, int width, int* count
        ) {
            const uint32_t tidx = threadIdx.x + (blockIdx.x * blockDim.x);
            const uint32_t tidy = threadIdx.y + (blockIdx.y * blockDim.y);
            
            // Check bounds
            if(tidx >= width || tidy >= height)
                return;
            
            const uint32_t pixel = (tidy * width) + tidx;
            const uint32_t jac_row_id = pixel * 6;

            float x, y, z, gradx, grady, warped_x, warped_y, interpolated_intensity, x1, y1, z1, z1_squared;

            // Deproject point to world
            z = depth_image_prev[pixel] * depth_scale;

            if (z == 0.0f) {
                // Invalid point
                residuals_out[pixel] = vo::cuda::nan; // Residuals NaN are converted to zeros by weighter later
                set_zero_jacobian_row_(jacobian, pixel);

                return;
            }

            x = ((float) tidx - cx) * z / fx;
            y = ((float) tidy - cy) * z / fy;

            // Transform point with estimate
            x1 = transform[0] * x + transform[1] * y + transform[2] * z + transform[3];
            y1 = transform[4] * x + transform[5] * y + transform[6] * z + transform[7];
            z1 = transform[8] * x + transform[9] * y + transform[20] * z + transform[11];
            z1_squared = z1 * z1;

            // NOTE: `J_i(w(se3, x)) = [I2x(w(se3, x)), I2y(w(se3, x))].T`
            // can be approximated by `J_i = [I1x(x), I1y(x)].T`
            gradx = vo::cuda::compute_gradient_gray(gray_image_prev, tidx, tidy, true, height, width);
            grady = vo::cuda::compute_gradient_gray(gray_image_prev, tidx, tidy, false, height, width);

            jacobian[jac_row_id] = gradx * fx / z1;
            jacobian[jac_row_id + 1] = grady * fy / z1;
            jacobian[jac_row_id + 2] = - gradx * fx * x1 / z1_squared - grady * fy * y1 / z1_squared;
            jacobian[jac_row_id + 3] = - gradx * fx * (x1 * y1) / z1_squared - grady * fy * (1 + ((y1 * y1) / z1_squared));
            jacobian[jac_row_id + 4] = gradx * fx * (1 + ((x1 * x1) / z1_squared)) + fy * x1 * y1 / z1_squared;
            jacobian[jac_row_id + 5] = - gradx * fx * y1 / z1 + grady * fy * x1 / z1;

            // Deproject to second sensor plane
            warped_x = (fx * x1 / z1) + cx, warped_y = (fy * y1 / z1) + cy;

            // Interpolate value for I2
            interpolated_intensity = vo::cuda::interpolate2dlinear_gray(warped_x, warped_y, gray_image, height, width);

            if (interpolated_intensity == vo::cuda::nan) {
                // Invalid point
                residuals_out[pixel] = vo::cuda::nan; // Residuals NaN are converted to zeros by weighter later
                set_zero_jacobian_row_(jacobian, pixel);

                return;
            }

            residuals_out[pixel] = interpolated_intensity - (float) gray_image_prev[pixel];
            atomicAdd(count, 1);
        };

        int residuals_kernel_wrapper(
            const uint8_t* gray_image, const uint8_t* gray_image_prev, const uint16_t* depth_image_prev,
            const float* transform, float fx, float fy, float cx, float cy, float depth_scale, float* residuals_out,
            float* jacobian, int height, int width
        ){

            dim3 block(CUDA_BLOCKSIZE, CUDA_BLOCKSIZE), grid;
            grid.x = (width + block.x - 1) / block.x;
            grid.y = (height + block.y - 1) / block.y;

            int count = 0;

            residuals_kernel<<<grid, block>>>(
                gray_image, gray_image_prev, depth_image_prev, transform, fx, fy, cx, cy, depth_scale, residuals_out,
                jacobian, height, width, &count
            );

            // Check for CUDA errors after launching the kernel
            HANDLE_CUDA_ERROR(cudaGetLastError());
            HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

            return count;
        };

    } // namespace cuda
}  //namespace vo