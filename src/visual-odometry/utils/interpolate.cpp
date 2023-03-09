#include <utils/interpolate.h>

namespace vo {
    namespace util {
        float interpolate2dlinear(const float x, const float y, const cv::Mat& gray_image){
            int x0 = floor(x), y0 = floor(y);
            int x1 = x0 + 1, y1 = y0 + 1;

            // Return nan if coordiante lies outside of image grid
            if ((x0 < 0) || (y0 < 0) || (x1 >= gray_image.cols) || (y1 >= gray_image.rows)){
                return vo::util::nan;
            }

            float w00 = (x1 - x) * (y1 - y), w01 = (x1 - x) * (y - y0), w10 = (x - x0) * (y1 - y), w11 = (x - x0) * (y - y0);

            float interpolated_value = (
                (w00 * gray_image.at<uint8_t>(y0, x0) + w01 * gray_image.at<uint8_t>(y1, x0) +
                w10 * gray_image.at<uint8_t>(y0, x1) + w11 * gray_image.at<uint8_t>(y1, x1)) /
                ((x1 - x0) * (y1 - y0))
            );

            return interpolated_value;
        }
    }
}
