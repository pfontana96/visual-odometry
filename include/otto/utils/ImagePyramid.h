#ifndef IMAGE_PYRAMID_H
#define IMAGE_PYRAMID_H

#include <vector>
#include <iterator>
#include <cstddef>
#include <string>
#include <cassert>
#include <iostream>

#include <opencv4/opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace otto {
    class RGBDImagePyramid {
        public:
            // Methods
            RGBDImagePyramid(const cv::Mat& gray_image, const cv::Mat& depth_image, const int levels);
            ~RGBDImagePyramid();

            inline cv::Mat get_gray_level(int level) {
                assert((
                    ("Got incorrect 'level': " + std::to_string(level) + " for pyramid with " + std::to_string(levels_) + " levels"),
                    (level>=0) && (level<levels_)
                ));
                return gray_pyramid_[level];
            }

            inline cv::Mat get_depth_level(int level) {
                assert((
                    ("Got incorrect 'level': " + std::to_string(level) + " for pyramid with " + std::to_string(levels_) + " levels"),
                    (level>=0) && (level<levels_)
                ));
                return depth_pyramid_[level];
            }

        private:
            // Attributes
            int current_level_, levels_;
            std::vector<cv::Mat> gray_pyramid_, depth_pyramid_;

            // Methods
            void build_pyramid();
    };
} // namespace otto

#endif