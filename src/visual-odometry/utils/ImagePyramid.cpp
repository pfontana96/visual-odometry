#include <utils/ImagePyramid.h>

namespace vo {
    namespace util {

        RGBDImagePyramid::RGBDImagePyramid(int levels):
            empty_(true),
            levels_(levels)
        {
            #ifdef VO_CUDA_ENABLED
            std::cout << "Yee haa CUDA ENABLED" << std::endl;
            #else
            std::cout << "NO CUDA neeeeddeeddd" << std::endl;
            #endif

            assert((levels > 0) || ([levels] {
                fprintf(stderr, "Expected 'levels' to be greater than 0, got '%d'\n", levels);
                return false; 
            }()));

            gray_pyramid_.reserve(levels_);
            depth_pyramid_.reserve(levels_);
            intrinsics_.reserve(levels_);

            #ifdef VO_CUDA_ENABLED
            std::cout << "hola" << std::endl;
            std::cout << "hssd" << std::endl;
            std::cout << "como" << std::endl;
            std::cout << "sdgs" << std::endl;
            std::cout << "estas" << std::endl;
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

            if (empty_ == false) {
                gray_pyramid_.clear();
                depth_pyramid_.clear();
                intrinsics_.clear();

                gray_pyramid_.reserve(levels_);
                depth_pyramid_.reserve(levels_);
                intrinsics_.reserve(levels_);

                #ifdef VO_CUDA_ENABLED
                gray_pyramid_gpu_.clear();
                depth_pyramid_gpu_.clear();

                gray_pyramid_gpu_.reserve(levels_);
                depth_pyramid_gpu_.reserve(levels_);
                #endif
            }

            intrinsics_.push_back(intrinsics);

            #ifndef VO_CUDA_ENABLED
            gray_pyramid_.push_back(gray_image);
            depth_pyramid_.push_back(depth_image);

            #else
            int height = gray_image.rows, width = gray_image.cols;

            gray_pyramid_gpu_.emplace_back(std::make_unique<vo::cuda::CudaSharedArray<uint8_t>>(height, width));
            depth_pyramid_gpu_.emplace_back(std::make_unique<vo::cuda::CudaSharedArray<uint16_t>>(height, width));
            std::cout << "ip start" << std::endl;

            gray_pyramid_.emplace_back(height, width, CV_8UC1, gray_pyramid_gpu_[0]->get());
            depth_pyramid_.emplace_back(height, width, CV_16UC1, depth_pyramid_gpu_[0]->get());

            gray_image.copyTo(gray_pyramid_[0]);
            depth_image.copyTo(depth_pyramid_[0]);
            #endif

            for(int i = 1; i < levels_; i++) {

                #ifndef VO_CUDA_ENABLED
                cv::Mat gray, depth;
                vo::util::pyrDownMedianSmooth<uint8_t>(gray_pyramid_[i - 1], gray);
                vo::util::pyrDownMedianSmooth<uint16_t>(depth_pyramid_[i - 1], depth);

                gray_pyramid_.push_back(gray);
                depth_pyramid_.push_back(depth);
                
                #else
                height = floor(height / 2);
                width = floor(width / 2);

                gray_pyramid_gpu_.emplace_back(std::make_unique<vo::cuda::CudaSharedArray<uint8_t>>(height, width));
                depth_pyramid_gpu_.emplace_back(std::make_unique<vo::cuda::CudaSharedArray<uint16_t>>(height, width));
                std::cout << "ip cpt 1 level" << i + 1 << std::endl;

                gray_pyramid_.emplace_back(height, width, CV_8UC1, gray_pyramid_gpu_[i]->get());
                depth_pyramid_.emplace_back(height, width, CV_16UC1, depth_pyramid_gpu_[i]->get());

                vo::util::pyrDownMedianSmooth<uint8_t>(gray_pyramid_[i - 1], gray_pyramid_[i]);
                vo::util::pyrDownMedianSmooth<uint16_t>(depth_pyramid_[i - 1], depth_pyramid_[i]);
                std::cout << "ip end level " << i+1 << std::endl;

                #endif

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
