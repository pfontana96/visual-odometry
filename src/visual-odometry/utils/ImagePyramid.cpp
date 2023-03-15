#include <utils/ImagePyramid.h>

namespace vo {
    namespace util {

        RGBDImagePyramid::RGBDImagePyramid(int levels):
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

            #ifdef VO_CUDA_ENABLED
            gray_pyramid_gpu_.reserve(levels_);
            depth_pyramid_gpu_.reserve(levels_);
            #endif
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

            int height = gray_image.rows, width = gray_image.cols;

            if (empty_) {
                #ifdef VO_CUDA_ENABLED
                gray_pyramid_gpu_.emplace_back(std::make_shared<vo::cuda::CudaArray<uint8_t>>(height, width, true));
                depth_pyramid_gpu_.emplace_back(std::make_shared<vo::cuda::CudaArray<uint16_t>>(height, width, true));

                gray_pyramid_.emplace_back(height, width, CV_8UC1, gray_pyramid_gpu_[0]->pointer.get());
                depth_pyramid_.emplace_back(height, width, CV_16UC1, depth_pyramid_gpu_[0]->pointer.get());

                #else
                gray_pyramid_.emplace_back(height, width, CV_8UC1);
                depth_pyramid_.emplace_back(height, width, CV_16UC1);

                #endif
            
            } else {
                // @note currently not bothering on optimizing the creation ofs everal intrinsics matrix as it's small
                intrinsics_.clear();
                intrinsics_.reserve(levels_);
            }

            gray_image.copyTo(gray_pyramid_[0]);
            depth_image.copyTo(depth_pyramid_[0]);

            intrinsics_.push_back(intrinsics);

            for(int i = 1; i < levels_; i++) {

                height = floor(height / 2);
                width = floor(width / 2);

                if(empty_) {
                    #ifdef VO_CUDA_ENABLED
                    gray_pyramid_gpu_.emplace_back(std::make_shared<vo::cuda::CudaArray<uint8_t>>(height, width, true));
                    depth_pyramid_gpu_.emplace_back(std::make_shared<vo::cuda::CudaArray<uint16_t>>(height, width, true));

                    gray_pyramid_.emplace_back(height, width, CV_8UC1, gray_pyramid_gpu_[i]->pointer.get());
                    depth_pyramid_.emplace_back(height, width, CV_16UC1, depth_pyramid_gpu_[i]->pointer.get());

                    #else
                    gray_pyramid_.emplace_back(height, width, CV_8UC1);
                    depth_pyramid_.emplace_back(height, width, CV_16UC1);

                    #endif
                }

                vo::util::pyrDownMedianSmooth<uint8_t>(gray_pyramid_[i - 1], gray_pyramid_[i]);
                vo::util::pyrDownMedianSmooth<uint16_t>(depth_pyramid_[i - 1], depth_pyramid_[i]);

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
