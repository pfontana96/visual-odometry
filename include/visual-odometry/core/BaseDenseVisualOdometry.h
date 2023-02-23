#ifndef VO_BASE_DENSE_DVO
#define VO_BASE_DENSE_DVO

#include <iostream>
#include <cassert>
#include <limits>
#include <string>
#include <exception>
#include <iostream>
// #include <filesystem>

#include <opencv2/core.hpp>

#include <sophus/se3.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <utils/types.h>
#include <utils/ImagePyramid.h>

#include <weighter/BaseWeighter.h>
#include <weighter/TDistributionWeighter.h>
#include <weighter/UniformWeighter.h>


namespace vo {
    namespace core {

        class BaseDenseVisualOdometry {

            public:

                BaseDenseVisualOdometry(
                    const int levels, const bool use_gpu, const bool use_weighter, const float sigma,
                    const int max_iterations, const float tolerance
                );
                virtual ~BaseDenseVisualOdometry();

                vo::util::Mat4f step(
                    const cv::Mat& color_image, const cv::Mat &depth_image,
                    const Eigen::Ref<const vo::util::Mat4f> init_guess
                );

                inline void update_camera_info(
                    Eigen::Ref<const vo::util::Mat3f> new_camera_intrinsics, const int new_height, const int new_width,
                    const float new_depth_scale
                ) {
                    height_ = new_height;
                    width_ = new_width;
                    intrinsics_ = new_camera_intrinsics;
                    first_frame_ = true;
                    no_camera_info_ = false;
                    depth_scale_ = new_depth_scale;
                }

            private:
                // Attributes
                vo::util::RGBDImagePyramid last_rgbd_pyramid_, current_rgbd_pyramid_;

                int levels_, max_iterations_, height_, width_;
                float tolerance_, sigma_;
                bool use_gpu_, use_weighter_, first_frame_, no_camera_info_;

                vo::util::Mat3f intrinsics_;
                float depth_scale_;

                vo::util::Mat4f last_estimate_;

                // Methods
                void non_linear_least_squares_(vo::util::Mat4f& xi, const int level);

                virtual int compute_residuals_and_jacobian_(
                    const cv::Mat& gray_image, const cv::Mat& gray_image_prev,
                    const cv::Mat& depth_image_prev, Eigen::Ref<const vo::util::Mat4f> transform,
                    const Eigen::Ref<const vo::util::Mat3f> intrinsics, const float depth_scale,
                    cv::Mat& residuals_out, vo::util::MatX6f& jacobian
                ) = 0;

                inline void update_last_pyramid() {
                    last_rgbd_pyramid_.update(current_rgbd_pyramid_);
                }
        };
    } // namespace core
} // namespace vo

#endif