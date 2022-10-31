#include <otto/utils/ImagePyramid.h>

namespace otto {
    RGBDImagePyramid::RGBDImagePyramid(const cv::Mat& gray_image, const cv::Mat& depth_image, const int levels): 
        levels_(levels),
        current_level_(0)
    {
        assert (("Expected 'levels' to be greater than 0.", levels > 0));
        gray_pyramid_.push_back(gray_image_);
        depth_pyramid_.push_back(depth_image_);

        build_pyramid();
    }

    RGBDImagePyramid::~RGBDImagePyramid() {}

    void RGBDImagePyramid::build_pyramid() {
        for(size_t i = 1; i < levels_; i++) {
            cv::Mat gray_prev = gray_pyramid_[i - 1], depth_prev = depth_pyramid_[i - 1], gray, depth;
            cv::pyrDown(gray_prev, gray);
            cv::pyrDown(depth_prev, depth);
            gray_pyramid_.push_back(gray);
            depth_pyramid_.push_back(depth);
        }
    }


} // namespace otto