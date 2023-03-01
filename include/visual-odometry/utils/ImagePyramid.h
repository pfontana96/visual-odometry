#ifndef IMAGE_PYRAMID_H
#define IMAGE_PYRAMID_H

#include <vector>
#include <iterator>
#include <cstddef>
#include <string>
#include <cassert>
#include <iostream>
#include <cmath>
#include <exception>

#include <opencv4/opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <utils/types.h>


namespace vo {
    namespace util {

        // methods
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

        // BaseRGBDImagePyramid
        class BaseRGBDImagePyramid {
            public:
                BaseRGBDImagePyramid(int levels);
                virtual ~BaseRGBDImagePyramid();

                virtual void build_pyramids(
                    const cv::Mat& gray_image, const cv::Mat& depth_image,
                    const Eigen::Ref<const vo::util::Mat3f> intrinsics
                ) = 0;

                inline void update(const vo::util::BaseRGBDImagePyramid& other) {

                    cv::Mat gray = other.gray_at(0).clone(), depth = other.depth_at(0).clone();
                    vo::util::Mat3f intrinsics = other.intrinsics_at(0);

                    build_pyramids(gray, depth, intrinsics);
                };

                inline cv::Mat gray_at(int level) const {
                    assert((
                        ("Got incorrect 'level': " + std::to_string(level) + " for pyramid with " + std::to_string(levels_) + " levels"),
                        (level>=0) && (level<levels_)
                    ));
                    assert(("Cannot query empty pyramid", empty_ != true));

                    return gray_pyramid_[level];
                };

                inline cv::Mat depth_at(int level) const {
                    assert((
                        ("Got incorrect 'level': " + std::to_string(level) + " for pyramid with " + std::to_string(levels_) + " levels"),
                        (level>=0) && (level<levels_)
                    ));
                    assert(("Cannot query empty pyramid", empty_ != true));

                    return depth_pyramid_[level];
                };

                inline vo::util::Mat3f intrinsics_at(int level) const {
                    assert((
                        ("Got incorrect 'level': " + std::to_string(level) + " for pyramid with " + std::to_string(levels_) + " levels"),
                        (level>=0) && (level<levels_)
                    ));
                    assert(("Cannot query empty pyramid", empty_ != true));

                    return intrinsics_[level];
                };

                inline bool empty() const {
                    return empty_;
                }

            protected:
                // Attributes
                bool empty_;
                int levels_;
                std::vector<cv::Mat> gray_pyramid_, depth_pyramid_;
                std::vector<vo::util::Mat3f> intrinsics_;
        };

        class RGBDImagePyramid : public BaseRGBDImagePyramid {
            public:
                // Methods
                RGBDImagePyramid(int levels);
                ~RGBDImagePyramid();

                void build_pyramids(
                    const cv::Mat& gray_image, const cv::Mat& depth_image,
                    const Eigen::Ref<const vo::util::Mat3f> intrinsics
                ) override;
        };
    }  // namespace utils
} // namespace vo

#endif