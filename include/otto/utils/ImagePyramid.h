#ifndef IMAGE_PYRAMID_H
#define IMAGE_PYRAMID_H

#include <vector>
#include <iterator>
#include <cstddef>

#include <opencv4/opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace otto {
    class RGBDImagePyramid {
        public:
            // Methods
            RGBDImagePyramid(const cv::Mat& gray_image, const cv::Mat& depth_image, const int levels);
            ~RGBDImagePyramid();

            inline std::vector<cv::Mat> get_gray_pyramid() {
                return gray_pyramid_;
            }

            inline std::vector<cv::Mat> get_depth_pyramid() {
                return depth_pyramid_;
            }

        private:
            // Attributes
            int current_level_, levels_;
            std::vector<cv::Mat> gray_pyramid_, depth_pyramid_;
            cv::Mat gray_image_, depth_image_;

            // Methods
            void build_pyramid();
    };
} // namespace otto

#endif