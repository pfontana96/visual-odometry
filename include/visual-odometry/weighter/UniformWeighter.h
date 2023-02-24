#ifndef VO_WEIGHTER_UNIF_H
#define VO_WEIGHTER_UNIF_H

#include <opencv2/core.hpp>

#include <weighter/BaseWeighter.h>


namespace vo {
    namespace weighter {
        // Dummy class equivalent to no weighting
        class UniformWeighter : public BaseWeighter {
            public:
                UniformWeighter();
                ~UniformWeighter();

                float weight(cv::Mat& residuals, cv::Mat& weights_out) override;
        };
    } // namespace weighter
} // namespace vo


#endif