#include <opencv2/core.hpp>

#include <utils/ImagePyramid.h>

extern "C" {
    namespace ottopy {
        otto::RGBDImagePyramid* RGBDImagePyramid_new(const cv::Mat& gray_image, const cv::Mat& depth_image, const int levels) {
            return new otto::RGBDImagePyramid(gray_image, depth_image, levels);
        };
    }
}   