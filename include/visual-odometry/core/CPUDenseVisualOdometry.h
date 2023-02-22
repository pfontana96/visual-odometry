#ifndef VO_DVO_H
#define VO_DVO_H

#include <cmath>

#include <core/BaseDenseVisualOdometry.h>
#include <utils/interpolate.h>
#include <utils/YAMLLoader.h>


namespace vo {
    namespace core {

        class CPUDenseVisualOdometry : public BaseDenseVisualOdometry {

            public:

                CPUDenseVisualOdometry(
                    const int levels, const bool use_gpu, const bool use_weighter, const float sigma,
                    const int max_iterations, const float tolerance
                );
                ~CPUDenseVisualOdometry();

                void compute_residuals_and_jacobian_(
                    const cv::Mat& gray_image, const cv::Mat& gray_image_prev,
                    const cv::Mat& depth_image_prev, Eigen::Ref<const vo::util::Mat4f> transform,
                    const Eigen::Ref<const vo::util::Mat3f> intrinsics, const float depth_scale,
                    cv::Mat& residuals_out, vo::util::MatX6f& jacobian,
                    vo::util::NonLinearLeastSquaresSolver& solver
                ) override;

                static vo::core::CPUDenseVisualOdometry load_from_yaml(const std::string filename);

                template<typename T>
                static float compute_gradient(const cv::Mat& image, const int x, const int y, const bool x_direction) {
                    
                    T prev_value, next_value;

                    if (x_direction == true) {
                        prev_value = image.at<T>(y, std::max<int>(x - 1, 0));
                        next_value = image.at<T>(y, std::min<int>(x + 1, image.cols - 1));
                    } else {
                        prev_value = image.at<T>(std::max<int>(y - 1, 0), x);
                        next_value = image.at<T>(std::min<int>(y + 1, image.rows - 1), x);
                    }

                    return 0.5f * (next_value - prev_value);
                };

        };
    } // namespace core
} // namespace vo


#endif
