#ifndef CAMERA_MODEL_H
#define CAMERA_MODEL_H

#include <iostream>
#include <cassert>

#include <opencv2/core.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <utils/types.h>

namespace otto {

    class RGBDCamera {
        public:
            RGBDCamera(const Mat3f& camera_intrinsics, const float depth_scale);
            deproject(const cv::Mat& depth_image, const Mat4f& camera_pose)
        private:
            cv::Mat gray_image_prev_, depth_image_prev_;
    }

}  // namespace otto

#endif