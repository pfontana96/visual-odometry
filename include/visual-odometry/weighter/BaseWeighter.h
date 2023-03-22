#ifndef VO_WEIGHTER_BASE_H
#define VO_WEIGHTER_BASE_H

#include <opencv2/core.hpp>

#include <utils/common.h>


namespace vo {
    namespace weighter {
        class BaseWeighter {
            public:
                BaseWeighter();
                virtual ~BaseWeighter();

                // Any weighter should update NaNs to 0.0f
                virtual float weight(cv::Mat& residuals, cv::Mat& weights_out) = 0;
        };
    } // namespace weighter
} // namespace vo


#endif