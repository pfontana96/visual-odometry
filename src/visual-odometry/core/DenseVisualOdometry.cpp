#include <core/DenseVisualOdometry.h>

namespace vo {
    namespace core {
        DenseVisualOdometry::DenseVisualOdometry(
            int levels, bool use_gpu, bool use_weighter, float sigma, int max_iterations, float tolerance
        ):
            levels_(levels),
            use_gpu_(use_gpu),
            use_weighter_(use_weighter),
            sigma_(sigma),
            tolerance_(tolerance),
            max_iterations_(max_iterations),
            last_rgbd_pyramid_(levels),
            current_rgbd_pyramid_(levels),
            first_frame_(true),
            no_camera_info_(true)
        {
            if (use_weighter_){
                weighter_ = std::make_shared<vo::weighter::TDistributionWeighter>(5, 5.0, 0.001, 50);
                std::cout << "Using T-Distribution weighter" << std::endl;
            } else {
                weighter_ = std::make_shared<vo::weighter::UniformWeighter>();
                std::cout << "Not using weighter" << std::endl;
            }
        }

        DenseVisualOdometry::~DenseVisualOdometry(){};

        /**
         * @brief Loads Dense Visual Odometry from YAML config file
         * 
         * @param filename 
         * @return vo::core::DenseVisualOdometry 
         */
        vo::core::DenseVisualOdometry DenseVisualOdometry::load_from_yaml(const std::string filename) {

            vo::util::YAMLLoader loader(filename);

            // Load necessary parameters from YAML file
            int levels = loader.get_value<int>("levels");
            bool use_gpu = loader.get_value<bool>("use_gpu");
            bool use_weighter = loader.get_value<bool>("use_weighter");
            float sigma = loader.get_value<float>("sigma");
            int max_iterations = loader.get_value<int>("max_iterations");
            float tolerance = loader.get_value<float>("tolerance");

            return vo::core::DenseVisualOdometry(levels, use_gpu, use_weighter, sigma, max_iterations, tolerance);
        }

        /**
         * @brief Estimates the transform between the last frame and the current one.
         * 
         * @param color_image Color image of current frame.
         * @param depth_image Depth image of current frame.
         * @param init_guess Initial guess of the transform to estimate.
         * @return vo::util::Mat4f
         */
        vo::util::Mat4f DenseVisualOdometry::step(
            const cv::Mat& color_image, const cv::Mat &depth_image, const Eigen::Ref<const vo::util::Mat4f> init_guess
        ) {
            assert ((no_camera_info_ == false) && "No camera info set. Cannot do a step");

            cv::Mat gray_image;
            cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);

            if (first_frame_ == true) {

                last_rgbd_pyramid_.build_pyramids(
                    gray_image, depth_image, intrinsics_
                );

                first_frame_ = false;
                vo::util::Mat4f transform = vo::util::Mat4f::Identity();
                last_estimate_ = transform;
                
                return transform;
            }

            current_rgbd_pyramid_.build_pyramids(
                gray_image, depth_image, intrinsics_
            );

            vo::util::Mat4f estimate = init_guess;

            for (int i = levels_ - 1; i >= 0; i--){
                non_linear_least_squares_(estimate, i);
            }

            // Update
            update_last_pyramid();
            last_estimate_ = estimate;

            return estimate;
        }

        void DenseVisualOdometry::non_linear_least_squares_(Eigen::Ref<vo::util::Mat4f> estimate, int level) {

            float error_prev = std::numeric_limits<float>::max();

            cv::Size cv_size = current_rgbd_pyramid_.gray_at(level).size();
            cv::Mat residuals_image(cv_size.height, cv_size.width, CV_32F);
            Eigen::Map<vo::util::VecXf> residuals(
                residuals_image.ptr<float>(), cv_size.height * cv_size.width
            );

            cv::Mat weights_image(cv_size.height, cv_size.width, CV_32F);
            Eigen::Map<vo::util::VecXf> weights(
                weights_image.ptr<float>(), cv_size.height * cv_size.width
            );

            vo::util::MatX6f jacobian(cv_size.height * cv_size.width, 6);
            vo::util::Vec6f solution, b;
            vo::util::Mat6f H;

            Sophus::SE3f increment, old(last_estimate_), xi(estimate);

            float count, error, error_diff;

            std::string out_message;

            for (size_t it = 0; it < (size_t) max_iterations_; it++) {

                count = compute_residuals_and_jacobian_(
                    current_rgbd_pyramid_.gray_at(level), last_rgbd_pyramid_.gray_at(level),
                    last_rgbd_pyramid_.depth_at(level), xi.matrix(), last_rgbd_pyramid_.intrinsics_at(level),
                    depth_scale_, residuals_image, jacobian
                );

                // Solve Normal equations
                error = weighter_->weight(residuals_image, weights_image);
                // error /= (float) count;

                H = jacobian.transpose() * weights.asDiagonal() * jacobian;
                b = - (jacobian.transpose() * residuals.cwiseProduct(weights));

                if (sigma_ > 0.0f) {
                    H += ((1 / sigma_) * vo::util::Mat6f::Identity());
                    b += ((1 / sigma_) * old.log());
                }

                solution = H.ldlt().solve(b);

                error_diff = (error - error_prev);

                if ( error_diff < 0.0) {
                    // Error decrease so update estimate
                    increment = Sophus::SE3f::exp(solution.cast<float>());
                    xi = increment * xi;

                    if (sigma_ > 0.0f) {
                        old = increment.inverse() * old;
                    }

                    // https://en.wikipedia.org/wiki/Non-linear_least_squares#Convergence_criteria
                    if (abs(error_diff / error_prev) < tolerance_){
                            out_message = "Iteration '" + std::to_string(it + 1) + "' (error: " +
                            std::to_string(error) + ") -> Found convergence";
                            break;
                    }

                } else {
                    out_message = "Iteration '" + std::to_string(it + 1) + "' (error: " +
                    std::to_string(error) + ") -> Error increased";
                    break;
                }

                if (it == ((size_t) max_iterations_ - 1))
                    out_message = "Iteration '" + std::to_string(it + 1) + "' (error: " + std::to_string(error) +
                    ") -> Exceeded maximum number of iterations";

                error_prev = error;
            }

            estimate = xi.matrix();

            vo::util::Vec6f cov = estimate_covariance_(H, error, count);

            std::cout << out_message << std::endl;

        }

        int DenseVisualOdometry::compute_residuals_and_jacobian_(
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
                    gradx = vo::core::DenseVisualOdometry::compute_gradient_<uint8_t>(gray_image_prev, u, v, true);
                    grady = vo::core::DenseVisualOdometry::compute_gradient_<uint8_t>(gray_image_prev, u, v, false);

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
    } // core
} // vo
