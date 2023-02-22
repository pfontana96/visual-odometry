#include <core/CPUDenseVisualOdometry.h>


namespace vo {
    namespace core {

        CPUDenseVisualOdometry::CPUDenseVisualOdometry(
            const int levels, const bool use_gpu, const bool use_weighter, const float sigma,
            const int max_iterations, const float tolerance
        ) :
            BaseDenseVisualOdometry(levels, use_gpu, use_weighter, sigma, max_iterations, tolerance)
        {}

        CPUDenseVisualOdometry::~CPUDenseVisualOdometry(){}

        /**
         * @brief Loads Dense Visual Odometry from YAML config file
         * 
         * @param filename 
         * @return vo::core::CPUDenseVisualOdometry 
         */
        vo::core::CPUDenseVisualOdometry CPUDenseVisualOdometry::load_from_yaml(const std::string filename) {

            vo::util::YAMLLoader loader(filename);

            // Load necessary parameters from YAML file
            int levels = loader.get_value<int>("levels");
            bool use_gpu = loader.get_value<bool>("use_gpu");
            bool use_weighter = loader.get_value<bool>("use_weighter");
            float sigma = loader.get_value<float>("sigma");
            int max_iterations = loader.get_value<int>("max_iterations");
            float tolerance = loader.get_value<float>("tolerance");

            return vo::core::CPUDenseVisualOdometry(levels, use_gpu, use_weighter, sigma, max_iterations, tolerance);
        }

        void CPUDenseVisualOdometry::compute_residuals_and_jacobian_(
            const cv::Mat& gray_image, const cv::Mat& gray_image_prev,
            const cv::Mat& depth_image_prev, Eigen::Ref<const vo::util::Mat4f> transform,
            const Eigen::Ref<const vo::util::Mat3f> intrinsics, const float depth_scale,
            cv::Mat& residuals_out, vo::util::MatX6f& jacobian,
            vo::util::NonLinearLeastSquaresSolver& solver
        ) {

            float fx = intrinsics(0, 0), fy = intrinsics(1, 1), cx = intrinsics(0, 2), cy = intrinsics(1, 2);

            #pragma omp parallel for collapse(2)
            for (size_t v = 0; v < gray_image.rows; v++) {
                for (size_t u = 0; u < gray_image.cols; u++) {

                    size_t jac_row_id = u + v * gray_image.cols;

                    // Deproject point to world
                    float z = depth_image_prev.at<uint16_t>(v, u) * depth_scale;

                    if (z == 0.0f) {
                        // Invalid point
                        residuals_out.at<float>(v, u) = 0.0f;
                        jacobian.row(jac_row_id) << vo::util::nan, vo::util::nan, vo::util::nan,
                                                    vo::util::nan, vo::util::nan, vo::util::nan;

                        continue;
                    }

                    float x = (u - cx) * z / fx, y = (v - cy) * z / fy;

                    // Transform point with estimate
                    float x1 = transform(0, 0) * x + transform(0, 1) * y + transform(0, 2) * z + transform(0, 3);
                    float y1 = transform(1, 0) * x + transform(1, 1) * y + transform(1, 2) * z + transform(1, 3);
                    float z1 = transform(2, 0) * x + transform(2, 1) * y + transform(2, 2) * z + transform(2, 3);

                    // Compute Jacobian
                    // NOTE: `J_i(w(se3, x)) = [I2x(w(se3, x)), I2y(w(se3, x))].T`
                    // can be approximated by `J_i = [I1x(x), I1y(x)].T`
                    float gradx = vo::core::CPUDenseVisualOdometry::compute_gradient<uint8_t>(
                        gray_image_prev, u, v, true
                    );
                    float grady = vo::core::CPUDenseVisualOdometry::compute_gradient<uint8_t>(
                        gray_image_prev, u, v, false
                    );

                    jacobian.row(jac_row_id) << fx * gradx / z1,
                                                fy * grady / z1,
                                                -fx * gradx * x1 / (z1 * z1) - fy * grady * y1 / (z1 * z1),
                                                -fx * gradx * x1 * y1 / (z1 * z1) - fy * grady * (((y1 * y1) / (z1 * z1)) + 1),
                                                fx * gradx * (((x1 * x1)/(z1 * z1)) + 1) + fy * grady * x1 * y1 / (z1 * z1),
                                                -fx * gradx * y1 / z1 + fy * grady * x1 / z1;

                    // Deproject to second sensor plane
                    float warped_x = fx * x1 / z1 + cx, warped_y = fy * y1 / z1 + cy;

                    // Interpolate value for I2
                    float interpolated_intensity = vo::util::interpolate2dlinear(warped_x, warped_y, gray_image);

                    if (!std::isfinite(interpolated_intensity)) {
                        // Invalid point
                        residuals_out.at<float>(v, u) = 0.0f;
                        jacobian.row(jac_row_id) << vo::util::nan, vo::util::nan, vo::util::nan,
                                                    vo::util::nan, vo::util::nan, vo::util::nan;

                        continue;
                    }

                    residuals_out.at<float>(v, u) = interpolated_intensity - gray_image_prev.at<uint8_t>(v, u);

                    // Update normal equations
                    solver.update(jacobian.row(jac_row_id), residuals_out.at<float>(v, u));
                }
            }
        }
    }
}
