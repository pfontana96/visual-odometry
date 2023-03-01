#ifndef VO_CUDA_IMAGE_PYRAMID_H
#define VO_CUDA_IMAGE_PYRAMID_H

#include <vector>
#include <cassert>
#include <iostream>
#include <exception>
#include <memory>
#include <type_traits>

#include <cuda/common.cuh>

#include <opencv4/opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <utils/types.h>
#include <utils/ImagePyramid.h>


namespace vo {
    namespace util {

        class RGBDImagePyramidGPU : public vo::util::BaseRGBDImagePyramid{
            public:
                // Methods
                RGBDImagePyramidGPU(const int levels);
                ~RGBDImagePyramidGPU();

                void build_pyramids(
                    const cv::Mat& gray_image, const cv::Mat& depth_image,
                    const Eigen::Ref<const vo::util::Mat3f> intrinsics
                ) override;

                inline uint8_t* gray_gpu_ptr_at(int level) const {
                    assert((
                        ("Got incorrect 'level': " + std::to_string(level) + " for pyramid with " + std::to_string(levels_) + " levels"),
                        (level>=0) && (level<levels_)
                    ));
                    assert(("Cannot query empty pyramid", empty_ != true));

                    return gray_pyramid_gpu_[level].data();
                };

                inline uint16_t* depth_gpu_ptr_at(int level) const {
                    assert((
                        ("Got incorrect 'level': " + std::to_string(level) + " for pyramid with " + std::to_string(levels_) + " levels"),
                        (level>=0) && (level<levels_)
                    ));
                    assert(("Cannot query empty pyramid", empty_ != true));

                    return depth_pyramid_gpu_[level].data();
                };

                inline float* intrinsics_gpu_ptr_at(int level) const {
                    assert((
                        ("Got incorrect 'level': " + std::to_string(level) + " for pyramid with " + std::to_string(levels_) + " levels"),
                        (level>=0) && (level<levels_)
                    ));
                    assert(("Cannot query empty pyramid", empty_ != true));

                    return intrinsics_gpu_[level].data();
                };

            private:
                std::vector<vo::cuda::CudaSharedArray<uint8_t>> gray_pyramid_gpu_;
                std::vector<vo::cuda::CudaSharedArray<uint16_t>> depth_pyramid_gpu_;
                std::vector<vo::cuda::CudaArray<float>> intrinsics_gpu_;
        };

    } // namespace util
} // namespace vo

#endif
