#ifndef VO_WEIGHTER_T_DIST_H
#define VO_WEIGHTER_T_DIST_H

#include <opencv2/core.hpp>

#include <weighter/BaseWeighter.h>


namespace vo {
    namespace weighter {

        class TDistributionWeighter : public BaseWeighter {
            public:
                // Methods
                TDistributionWeighter(
                    const int dof, const float initial_sigma, const float tolerance, const int max_iterations
                );
                ~TDistributionWeighter();

                float weight(cv::Mat& residuals, cv::Mat& weights_out) override;

            private:
                // Attributes
                float max_iterations_, initial_sigma_, initial_lambda_, tolerance_;
                int dof_;

                // Methods
                float compute_scale_(const cv::Mat& residuals, const float last_lambda);
                float compute_weights_(cv::Mat& residuals, cv::Mat& weights_out, const float lambda);
        };
    } // namespace weighter
} // namespace vo

#endif