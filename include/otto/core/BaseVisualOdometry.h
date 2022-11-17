#ifndef BASE_DVO_H
#define BASE_DVO_H

#include <iostream>
#include <cassert>

#include <opencv2/core.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <utils/types.h>

namespace otto {

    class RGBDBaseVisualOdometry {
        public:

        private:
            cv::Mat gray_image_prev_, depth_image_prev_;
    }

}  // namespace otto

#endif