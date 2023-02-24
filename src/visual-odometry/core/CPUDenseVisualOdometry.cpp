#include <core/CPUDenseVisualOdometry.h>


namespace vo {
    namespace core {

        CPUDenseVisualOdometry::CPUDenseVisualOdometry(
            int levels, bool use_gpu, bool use_weighter, float sigma, int max_iterations, float tolerance
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

        int CPUDenseVisualOdometry::compute_residuals_and_jacobian_(
            const cv::Mat& gray_image, const cv::Mat& gray_image_prev,
            const cv::Mat& depth_image_prev, const Eigen::Ref<const vo::util::Mat4f> transform,
            const Eigen::Ref<const vo::util::Mat3f> intrinsics, float depth_scale,
            cv::Mat& residuals_out, Eigen::Ref<vo::util::MatX6f> jacobian
        ) {

            float fx = intrinsics(0, 0), fy = intrinsics(1, 1), cx = intrinsics(0, 2), cy = intrinsics(1, 2);
            float x, y, z, gradx, grady, warped_x, warped_y, interpolated_intensity;
            int count = 0;

            vo::util::Vec3f point;
            Eigen::Matrix<float, 2, 6, Eigen::RowMajor> Jw;
            Eigen::Matrix<float, 1, 2> Ji;

            // #pragma omp parallel for private(Jw, Ji) shared(count)
            for (size_t v = 0; v < gray_image.rows; v++) {
                for (size_t u = 0; u < gray_image.cols; u++) {

                    int jac_row_id = (int) u + (int) v * gray_image.cols;

                    // Deproject point to world
                    z = depth_image_prev.at<uint16_t>(v, u) * depth_scale;

                    if (z == 0.0f) {
                        // Invalid point
                        residuals_out.at<float>(v, u) = vo::util::nan; // Residuals NaN are converted to zeros by weighter later
                        jacobian.row(jac_row_id).setZero();

                        continue;
                    }

                    x = ((float) u - cx) * z / fx;
                    y = ((float) v - cy) * z / fy;
                    point = {x, y, z};

                    // Transform point with estimate
                    point = transform.topLeftCorner<3, 3>() * point + transform.topRightCorner<3, 1>();
                    x = point(0);
                    y = point(1);
                    z = point(2);

                    // NOTE: `J_i(w(se3, x)) = [I2x(w(se3, x)), I2y(w(se3, x))].T`
                    // can be approximated by `J_i = [I1x(x), I1y(x)].T`
                    gradx = vo::core::CPUDenseVisualOdometry::compute_gradient<uint8_t>(
                        gray_image_prev, u, v, true
                    );
                    grady = vo::core::CPUDenseVisualOdometry::compute_gradient<uint8_t>(
                        gray_image_prev, u, v, false
                    );

                    Jw << fx / z,    0.0f, - fx * x / (z * z),         - fx * (x * y) / (z * z), fx * (1 + ((x * x) / (z * z))), - fx * y / z,
                             0.0f, fy / z, - fy * y / (z * z), - fy * (1 + ((y * y) / (z * z))),           fy * x * y / (z * z),   fy * x / z;
                    Ji << gradx, grady;
                    
                    jacobian.row(jac_row_id) = Ji * Jw;

                    // Deproject to second sensor plane
                    warped_x = (fx * x / z) + cx, warped_y = (fy * y / z) + cy;

                    // Interpolate value for I2
                    interpolated_intensity = vo::util::interpolate2dlinear(warped_x, warped_y, gray_image);

                    if (!std::isfinite(interpolated_intensity)) {
                        // Invalid point
                        residuals_out.at<float>(v, u) = vo::util::nan; // Residuals NaN are converted to zeros by weighter later
                        jacobian.row(jac_row_id).setZero();

                        continue;
                    }

                    residuals_out.at<float>(v, u) = interpolated_intensity - (float) gray_image_prev.at<uint8_t>(v, u);
                    count++;
                }
            }

            return count;
        }
    } // namespace core
} // namespace vo
