#include <weighter/UniformWeighter.h>

namespace vo {
    namespace weighter {

        UniformWeighter::UniformWeighter(){}

        UniformWeighter::~UniformWeighter(){}

        float UniformWeighter::weight(cv::Mat& residuals, cv::Mat& weights_out) {
            
            float error = 0.0f, residual_i;
            for (int y = 0; y < residuals.rows; y++) {
                for (int x = 0; x < residuals.cols; x++) {
                    residual_i = residuals.at<float>(y, x);

                    if (std::isfinite(residual_i)) {
                        weights_out.at<float>(y, x) = 1.0f;
                        error += residuals.at<float>(y, x) * residuals.at<float>(y, x);
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