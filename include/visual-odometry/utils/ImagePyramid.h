#ifndef IMAGE_PYRAMID_H
#define IMAGE_PYRAMID_H

#include <vector>
#include <iterator>
#include <cstddef>
#include <string>
#include <cassert>
#include <iostream>
#include <cmath>
#include <stdexcept>

#include <opencv4/opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <utils/types.h>

#ifdef VO_CUDA_ENABLED
#include <cuda/common.cuh>
#include <memory>
#endif

namespace vo {
    namespace util {

        template<typename T>
        static void pyrDownMedianSmooth(const cv::Mat& in, cv::Mat& out)
        {
            out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

            cv::Mat in_smoothed;
            cv::medianBlur(in, in_smoothed, 3);

            // #pragma omp parallel for
            for(int y = 0; y < out.rows; ++y)
            {
                for(int x = 0; x < out.cols; ++x)
                {
                    out.at<T>(y, x) = in_smoothed.at<T>(y * 2, x * 2);
                }
            }
        };

        class RGBDImagePyramid {
            public:
                // Methods
                RGBDImagePyramid(int levels);
                ~RGBDImagePyramid();

                void build_pyramids(
                    const cv::Mat& gray_image, const cv::Mat& depth_image,
                    const Eigen::Ref<const vo::util::Mat3f> intrinsics
                );

                inline cv::Mat gray_at(int level) {
                    if (level < 0 || level >= levels_)
                        throw std::invalid_argument(
                            "Expected 'level' to be greater than 0 and less than '" + std::to_string(levels_) +
                            "', got '" + std::to_string(level) + "' instead."
                        );

                    if (empty_)
                        throw std::runtime_error("Cannot query empty pyramid");

                    return gray_pyramid_[level];
                }

                inline cv::Mat depth_at(int level) {
                    if (level < 0 || level >= levels_)
                        throw std::invalid_argument(
                            "Expected 'level' to be greater than 0 and less than '" + std::to_string(levels_) +
                            "', got '" + std::to_string(level) + "' instead."
                        );

                    if (empty_)
                        throw std::runtime_error("Cannot query empty pyramid");

                    return depth_pyramid_[level];
                }

                inline vo::util::Mat3f& intrinsics_at(int level) {
                    if (level < 0 || level >= levels_)
                        throw std::invalid_argument(
                            "Expected 'level' to be greater than 0 and less than '" + std::to_string(levels_) +
                            "', got '" + std::to_string(level) + "' instead."
                        );

                    if (empty_)
                        throw std::runtime_error("Cannot query empty pyramid");

                    return intrinsics_[level];
                }

                #ifdef VO_CUDA_ENABLED
                inline uint8_t* gray_gpu_at(int level) {
                    if (level < 0 || level >= levels_)
                        throw std::invalid_argument(
                            "Expected 'level' to be greater than 0 and less than '" + std::to_string(levels_) +
                            "', got '" + std::to_string(level) + "' instead."
                        );

                    if (empty_)
                        throw std::runtime_error("Cannot query empty pyramid");

                    return gray_pyramid_gpu_[level]->pointer.get();
                }

                inline uint16_t* depth_gpu_at(int level) {
                    if (level < 0 || level >= levels_)
                        throw std::invalid_argument(
                            "Expected 'level' to be greater than 0 and less than '" + std::to_string(levels_) +
                            "', got '" + std::to_string(level) + "' instead."
                        );

                    if (empty_)
                        throw std::runtime_error("Cannot query empty pyramid");

                    return depth_pyramid_gpu_[level]->pointer.get();
                }
                #endif

                inline bool empty() {
                    return empty_;
                }

                inline void update(vo::util::RGBDImagePyramid& other) {

                    cv::Mat gray = other.gray_at(0).clone(), depth = other.depth_at(0).clone();
                    vo::util::Mat3f intrinsics = other.intrinsics_at(0);

                    build_pyramids(gray, depth, intrinsics);
                }

            private:
                // Attributes
                bool empty_;
                int levels_;
                std::vector<cv::Mat> gray_pyramid_, depth_pyramid_;
                std::vector<vo::util::Mat3f> intrinsics_;   

                #ifdef VO_CUDA_ENABLED
                std::vector<std::shared_ptr<vo::cuda::CudaArray<uint8_t>>> gray_pyramid_gpu_;
                std::vector<std::shared_ptr<vo::cuda::CudaArray<uint16_t>>> depth_pyramid_gpu_;
                #endif             
        };
    }  // namespace utils
} // namespace vo

#endif