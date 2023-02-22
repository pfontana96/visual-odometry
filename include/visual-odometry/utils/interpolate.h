#ifndef UTIL_INTERPOLATE
#define UTIL_INTERPOLATE

#include <cmath>

#include <opencv4/opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <utils/types.h>


namespace vo {
    namespace util {
        float interpolate2dlinear(const float x, const float y, const cv::Mat& gray_image);
    }
}

#endif
