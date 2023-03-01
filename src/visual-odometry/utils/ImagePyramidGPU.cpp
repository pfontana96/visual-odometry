#include <utils/ImagePyramidGPU.h>

namespace vo {
    namespace util {

        RGBDImagePyramidGPU::RGBDImagePyramidGPU(int levels):
            vo::util::BaseRGBDImagePyramid(levels)
        {
            gray_pyramid_gpu_.reserve(levels_);
            depth_pyramid_gpu_.reserve(levels_);
            intrinsics_gpu_.reserve(levels_);
        }

        RGBDImagePyramidGPU::~RGBDImagePyramidGPU() {}

        void RGBDImagePyramidGPU::build_pyramids(
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

                gray_pyramid_gpu_.clear();
                depth_pyramid_gpu_.clear();
                intrinsics_gpu_.clear();

                gray_pyramid_gpu_.reserve(levels_);
                depth_pyramid_gpu_.reserve(levels_);
                intrinsics_gpu_.reserve(levels_);
            }

            int height = gray_image.rows, width = gray_image.cols;

            gray_pyramid_gpu_.emplace_back(height, width);
            gray_pyramid_.emplace_back(height, width, gray_pyramid_gpu_[0].size(), gray_pyramid_gpu_[0].data());
            gray_image.copyTo(gray_pyramid_[0]);

            depth_pyramid_gpu_.emplace_back(height, width);
            depth_pyramid_.emplace_back(height, width, depth_pyramid_gpu_[0].size(), depth_pyramid_gpu_[0].data());
            depth_image.copyTo(depth_pyramid_[0]);

            intrinsics_.push_back(intrinsics);
            intrinsics_gpu_.emplace_back(3, 3);
            intrinsics_gpu_[0].copyFromHost(intrinsics_[0].data());

            for(int i = 1; i < levels_; i++) {

                height = floor(gray_pyramid_[i - 1].rows / 2);
                width = floor(gray_pyramid_[i - 1].cols / 2);

                gray_pyramid_gpu_.emplace_back(height, width);
                gray_pyramid_.emplace_back(height, width, gray_pyramid_gpu_[i].size(), gray_pyramid_gpu_[i].data());
                vo::util::pyrDownMedianSmooth<uint8_t>(gray_pyramid_[i - 1], gray_pyramid_[i]);

                depth_pyramid_gpu_.emplace_back(height, width);
                depth_pyramid_.emplace_back(height, width, depth_pyramid_gpu_[i].size(), depth_pyramid_gpu_[i].data());
                vo::util::pyrDownMedianSmooth<uint16_t>(depth_pyramid_[i - 1], depth_pyramid_[i]);

                vo::util::Mat3f scale_matrix;
                scale_matrix << powf(2, -i), 0.0, powf(2, -i - 1) - 0.5,
                                0.0, powf(2, -i), powf(2, -i - 1) - 0.5,
                                0.0, 0.0, 1.0;
                intrinsics_.push_back(scale_matrix * intrinsics_[i - 1]);
                intrinsics_gpu_.emplace_back(3, 3);
                intrinsics_gpu_[i].copyFromHost(intrinsics_[i].data());
            }

            empty_ = false;
        }

    } // namespace util
} // namespace vo
