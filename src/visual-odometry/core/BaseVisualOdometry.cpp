#include <core/BaseDenseVisualOdometry.h>

namespace vo {
    namespace core {
        BaseDenseVisualOdometry::BaseDenseVisualOdometry(
            const int levels, const bool use_gpu, const bool use_weighter, const float sigma,
            const int max_iterations, const float tolerance
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
        {}

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

                last_rgbd_pyramid_.build_pyramids(
                    gray_image, depth_image, intrinsics_
                );

                first_frame_ = false;
                vo::util::Mat4f transform = vo::util::Mat4f::Identity();
                last_estimate_ = Sophus::SE3f(
                    transform.topLeftCorner<3, 3>(), transform.topRightCorner<3, 1>()
                );
                
                return transform;
            }

            current_rgbd_pyramid_.build_pyramids(
                gray_image, depth_image, intrinsics_
            );

            Sophus::SE3f estimate = Sophus::SE3f(
                init_guess.topLeftCorner<3, 3>(), init_guess.topRightCorner<3, 1>()
            );

            for (int i = levels_ - 1; i >= 0; i--){
                non_linear_least_squares_(estimate, i);
            }

            // Update
            update_last_pyramid();
            last_estimate_ = estimate;
            vo::util::Mat4f transform = estimate.matrix();

            return transform;
        }

        void BaseDenseVisualOdometry::non_linear_least_squares_(Sophus::SE3f& estimate, const int level) {

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

            vo::weighter::BaseWeighter *weighter_;
            if (use_weighter_){
                weighter_ = new vo::weighter::TDistributionWeighter(5, 5.0, 0.001, 50);
            } else {
                weighter_ = new vo::weighter::UniformWeighter();
            }

            vo::util::MatX6f jacobian(cv_size.height * cv_size.width, 6);
            vo::util::Vec6f solution, b;
            vo::util::Mat6f H;

            Sophus::SE3f increment;

            float count, error, error_diff;

            for (size_t it = 0; it < (size_t) max_iterations_; it++) {

                count = compute_residuals_and_jacobian_(
                    current_rgbd_pyramid_.gray_at(level), last_rgbd_pyramid_.gray_at(level),
                    last_rgbd_pyramid_.depth_at(level), estimate.matrix(), last_rgbd_pyramid_.intrinsics_at(level),
                    depth_scale_, residuals_image, jacobian
                );

                // Solve Normal equations
                error = weighter_->weight(residuals_image, weights_image);
                error /= count;

                H = jacobian.transpose() * weights.asDiagonal() * jacobian;
                b = -jacobian.transpose() * weights.asDiagonal() * residuals;

                solution = H.ldlt().solve(b);

                error_diff = (error - error_prev);

                if ( error_diff < 0.0) {
                    // Error decrease so update estimate
                    increment = Sophus::SE3f::exp(solution.cast<float>());
                    estimate = increment * estimate;

                    if (abs(error_diff) <= tolerance_) {
                        std::cout << "Found convergence at iteration '" << it
                        << "' (error: " << error << ")" << std::endl;
                        break;
                    }

                } else {
                    std::cout << "Error '" << error << "' increased at iteration '" << it << "'" << std::endl;
                    break;
                }

                if (it == ((size_t) max_iterations_ - 1)) {
                    std::cout << "Exceeded maximum number of iterations '" << max_iterations_ << "'" << std::endl;
                }

                error_prev = error;
            }

            delete weighter_;
        }
    } // core
} // vo
