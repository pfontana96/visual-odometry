#ifndef VO_CUDA_RESIDUALS_KERNEL_H
#define VO_CUDA_RESIDUALS_KERNEL_H

#include <cuda/common.cuh>


namespace vo {
    namespace cuda {

        int residuals_kernel_wrapper(
            const uint8_t* gray_image, const uint8_t* gray_image_prev, const uint16_t* depth_image_prev,
            const float* transform, const float* intrinsics, float depth_scale, float* residuals_out, float* jacobian,
            int height, int width
        );

    } // namespace cuda
} // namespace vo

#endif