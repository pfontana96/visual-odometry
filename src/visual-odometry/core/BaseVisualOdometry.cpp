#include <core/BaseDenseVisualOdometry.h>

namespace vo {
    namespace core {
        BaseDenseVisualOdometry::BaseDenseVisualOdometry(
            int levels, bool use_gpu, bool use_weighter, float sigma, int max_iterations, float tolerance
        ):
            levels_(levels),
            use_gpu_(use_gpu),
            use_weighter_(use_weighter),
            sigma_(sigma),
            tolerance_(tolerance),
            max_iterations_(max_iterations),
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

            if (use_gpu_){
                #ifndef VO_CUDA_ENABLED
                    throw std::invalid_argument("'use_gpu' set to true but vo was not compiled with CUDA support.")
                #endif
            }

            if (use_gpu_) {
                last_rgbd_pyramid_ = std::make_shared<vo::util::RGBDImagePyramidGPU>(levels_);
                current_rgbd_pyramid_ = std::make_shared<vo::util::RGBDImagePyramidGPU>(levels_);

            } else {
                last_rgbd_pyramid_ = std::make_shared<vo::util::RGBDImagePyramid>(levels_);
                current_rgbd_pyramid_ = std::make_shared<vo::util::RGBDImagePyramid>(levels_);
            }
        }

        BaseDenseVisualOdometry::~BaseDenseVisualOdometry(){};

        /**
         * @brief Estimates the transform between the last frame and the current one.
         * 
         * @param color_image Color image of current frame.
         * @param depth_image Depth image of current frame.
         * @param init_guess Initial guess of the transform to estimate.
         * @return vo::util::Mat4f
         */
        vo::util::Mat4f BaseDenseVisualOdometry::step(
            const cv::Mat& color_image, const cv::Mat &depth_image, const Eigen::Ref<const vo::util::Mat4f> init_guess
        ) {
            assert ((no_camera_info_ == false) && "No camera info set. Cannot do a step");

            cv::Mat gray_image;
            cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);

            if (first_frame_ == true) {

                last_rgbd_pyramid_->build_pyramids(
                    gray_image, depth_image, intrinsics_
                );

                first_frame_ = false;
                vo::util::Mat4f transform = vo::util::Mat4f::Identity();
                last_estimate_ = transform;
                
                return transform;
            }

            current_rgbd_pyramid_->build_pyramids(
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

        void BaseDenseVisualOdometry::non_linear_least_squares_(Eigen::Ref<vo::util::Mat4f> estimate, int level) {

            float error_prev = std::numeric_limits<float>::max();

            cv::Size cv_size = current_rgbd_pyramid_->gray_at(level).size();
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
                    current_rgbd_pyramid_->gray_at(level), last_rgbd_pyramid_->gray_at(level),
                    last_rgbd_pyramid_->depth_at(level), xi.matrix(), last_rgbd_pyramid_->intrinsics_at(level),
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
    } // core
} // vo
