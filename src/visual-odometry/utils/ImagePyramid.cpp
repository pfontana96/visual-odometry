#include <utils/ImagePyramid.h>

namespace vo {
    namespace util {

        RGBDImagePyramid::RGBDImagePyramid(const int levels):
            empty_(true),
            levels_(levels)
        {
            assert((levels > 0) || ([levels] {
                fprintf(stderr, "Expected 'levels' to be greater than 0, got '%d'\n", levels);
                return false; 
            }()));

            gray_pyramid_.reserve(levels_);
            depth_pyramid_.reserve(levels_);
            intrinsics_.reserve(levels_);
        }

        RGBDImagePyramid::~RGBDImagePyramid() {}

        void RGBDImagePyramid::build_pyramids(
            const cv::Mat& gray_image, const cv::Mat& depth_image,
            const Eigen::Ref<const vo::util::Mat3f> intrinsics
        ) {

            if (gray_image.size() != depth_image.size()) {
                throw std::invalid_argument(
                    "Expected 'gray_image' and 'depth_image' to be equal, got '(" + std::to_string(gray_image.rows) +
                    ", " + std::to_string(gray_image.cols) + "' and '(" + std::to_string(depth_image.rows) + ", " + 
                    std::to_string(depth_image.cols) + ")' respectively."
                );
            }

            if (empty_ == false) {
                gray_pyramid_.clear();
                depth_pyramid_.clear();
                intrinsics_.clear();

                gray_pyramid_.reserve(levels_);
                depth_pyramid_.reserve(levels_);
                intrinsics_.reserve(levels_);
            }

            gray_pyramid_.push_back(gray_image);
            depth_pyramid_.push_back(depth_image);
            intrinsics_.push_back(intrinsics);

            for(size_t i = 1; i < levels_; i++) {

                cv::Mat gray, depth;
                vo::util::pyrDownMedianSmooth<uint8_t>(gray_pyramid_[i - 1], gray);
                vo::util::pyrDownMedianSmooth<uint16_t>(depth_pyramid_[i - 1], depth);

                gray_pyramid_.push_back(gray);
                depth_pyramid_.push_back(depth);

                vo::util::Mat3f scale_matrix;
                scale_matrix << powf(2, -i), 0.0, powf(2, -i - 1) - 0.5,
                                0.0, powf(2, -i), powf(2, -i - 1) - 0.5,
                                0.0, 0.0, 1.0;
                intrinsics_.push_back(scale_matrix * intrinsics_[i - 1]);

            }

            empty_ = false;
        }
    } // namespace util
} // namespace vo
