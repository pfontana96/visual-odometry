#ifndef VO_BASE_DENSE_DVO
#define VO_BASE_DENSE_DVO

#include <iostream>
#include <cassert>
#include <limits>
#include <string>
#include <exception>
#include <iostream>
#include <memory>
// #include <filesystem>

#include <opencv2/core.hpp>

#include <sophus/se3.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <utils/types.h>
#include <utils/ImagePyramid.h>
#include <utils/interpolate.h>
#include <utils/YAMLLoader.h>

#include <weighter/BaseWeighter.h>
#include <weighter/TDistributionWeighter.h>
#include <weighter/UniformWeighter.h>


namespace vo {
    namespace core {

        class DenseVisualOdometry {

            public:

                DenseVisualOdometry(
                    int levels, bool use_gpu, bool use_weighter, float sigma = -1.0f, int max_iterations = 100,
                    float tolerance = 1e-5
                );
                ~DenseVisualOdometry();

                static vo::core::DenseVisualOdometry load_from_yaml(const std::string filename);

                vo::util::Mat4f step(
                    const cv::Mat& color_image, const cv::Mat &depth_image,
                    const Eigen::Ref<const vo::util::Mat4f> init_guess
                );

                inline void update_camera_info(
                    const Eigen::Ref<const vo::util::Mat3f> new_camera_intrinsics,
                    int new_height, int new_width, float new_depth_scale
                ) {
                    height_ = new_height;
                    width_ = new_width;
                    intrinsics_ = new_camera_intrinsics;
                    first_frame_ = true;
                    no_camera_info_ = false;
                    depth_scale_ = new_depth_scale;
                }

                inline void update_camera_info(
                    const std::string new_camera_intrinsics_filename
                ) {
                    vo::util::YAMLLoader loader(new_camera_intrinsics_filename);

                    int new_height = loader.get_value<int>("height");
                    int new_width = loader.get_value<int>("width");
                    vo::util::Mat3f new_camera_intrinsics = loader.get_value<vo::util::Mat3f>("intrinsics");
                    float new_depth_scale = loader.get_value<float>("depth_scale");

                    update_camera_info(new_camera_intrinsics, new_height, new_width, new_depth_scale);
                }

                /**
                 * @brief Resets estimation in case required from an external source
                 * (e.g. SLAM backend)
                 * 
                 */
                inline void reset() {
                    first_frame_ = true;
                };

            private:
                // Attributes
                vo::util::RGBDImagePyramid last_rgbd_pyramid_, current_rgbd_pyramid_;

                int levels_, max_iterations_, height_, width_;
                float tolerance_, sigma_;
                bool use_gpu_, use_weighter_, first_frame_, no_camera_info_;

                vo::util::Mat3f intrinsics_;
                float depth_scale_;

                vo::util::Mat4f last_estimate_;

                std::shared_ptr<vo::weighter::BaseWeighter> weighter_;

                // Methods
                template<typename T>
                static inline float compute_gradient_(const cv::Mat& image, int x, int y, bool x_direction) {
                    
                    T prev_value, next_value;

                    if (x_direction == true) {
                        prev_value = image.at<T>(y, std::max<int>(x - 1, 0));
                        next_value = image.at<T>(y, std::min<int>(x + 1, image.cols - 1));
                    } else {
                        prev_value = image.at<T>(std::max<int>(y - 1, 0), x);
                        next_value = image.at<T>(std::min<int>(y + 1, image.rows - 1), x);
                    }

                    return 0.5f * ((float) (next_value - prev_value));
                };

                void non_linear_least_squares_(Eigen::Ref<vo::util::Mat4f> xi, int level);

                int compute_residuals_and_jacobian_(
                    const cv::Mat& gray_image, const cv::Mat& gray_image_prev,
                    const cv::Mat& depth_image_prev, const Eigen::Ref<const vo::util::Mat4f> transform,
                    const Eigen::Ref<const vo::util::Mat3f> intrinsics, float depth_scale,
                    cv::Mat& residuals_out, Eigen::Ref<vo::util::MatX6f> jacobian
                );

                inline void update_last_pyramid() {
                    last_rgbd_pyramid_.update(current_rgbd_pyramid_);
                }

                // https://stats.stackexchange.com/questions/93316/parameter-uncertainty-after-non-linear-least-squares-estimation
                inline vo::util::Vec6f estimate_covariance_(
                    const Eigen::Ref<const vo::util::Mat6f> hessian, float error, int nb_observations
                ) {
                    vo::util::Mat6f covariance = hessian.inverse();
                    covariance *= error / ((float) (nb_observations - 6));

                    return covariance.diagonal();
                }

        };
    } // namespace core
} // namespace vo

#endif