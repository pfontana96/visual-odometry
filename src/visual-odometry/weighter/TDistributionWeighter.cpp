#include <weighter/TDistributionWeighter.h>
#include <iostream>

namespace vo {
    namespace weighter {

        TDistributionWeighter::TDistributionWeighter(
            const int dof, const float initial_sigma, const float tolerance, const int max_iterations
        ):
            dof_(dof),
            initial_sigma_(initial_sigma),
            tolerance_(tolerance),
            max_iterations_(max_iterations)
        {
            initial_lambda_ = 1.0f / (initial_sigma_ * initial_sigma_);
        }

        TDistributionWeighter::~TDistributionWeighter(){}

        float TDistributionWeighter::weight(cv::Mat& residuals, cv::Mat& weights_out) {
            
            float last_lambda = initial_lambda_, sigma_squared, current_lambda, error;
            for (size_t i = 0; i < (size_t) max_iterations_; i++) {

                sigma_squared = compute_scale_(residuals, last_lambda);

                current_lambda = 1.0f / sigma_squared;
                if (abs(current_lambda - last_lambda) <= tolerance_) {
                    break;
                }

                last_lambda = current_lambda;
            }

            error = compute_weights_(residuals, weights_out, current_lambda);

            return error;
        }

        float TDistributionWeighter::compute_scale_(const cv::Mat& residuals, const float last_lambda) {
            
            float sigma_squared = 0.0f, residual_squared_i, residual_i;
            int count = 0;
            for (int y = 0; y < residuals.rows; y++) {
                for (int x = 0; x < residuals.cols; x++) {

                    residual_i = residuals.at<float>(y, x);

                    if (std::isfinite(residual_i)) {
                        residual_squared_i = residual_i * residual_i;
                        sigma_squared += (
                            residual_squared_i * (((float) dof_ + 1) / ((float) dof_ + residual_squared_i * last_lambda))
                        );
                        count ++;
                    }
                }
            }

            return sigma_squared / ((float) count);
        }

        float TDistributionWeighter::compute_weights_(cv::Mat& residuals, cv::Mat& weights_out, const float lambda) {
            float residual_squared_i, residual_i, error;
            for (int y = 0; y < residuals.rows; y++) {
                for (int x = 0; x < residuals.cols; x++) {

                    residual_i = residuals.at<float>(y, x);

                    if (std::isfinite(residual_i)) {
                        residual_squared_i = residual_i * residual_i;
                        weights_out.at<float>(y, x) = (
                            residual_squared_i * (((float) dof_ + 1) / ((float) dof_ + residual_squared_i * lambda))
                        );

                        error += weights_out.at<float>(y, x) * residual_squared_i;

                    } else {
                        residuals.at<float>(y, x) = 0.0f;
                        weights_out.at<float>(y, x) = 0.0f;
                    }
                }
            }

            return error; 
        }
    } // namespace weighter
} // namespace vo