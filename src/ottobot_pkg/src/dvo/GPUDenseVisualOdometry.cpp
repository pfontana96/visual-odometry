#include <GPUDenseVisualOdometry.h>

namespace otto{
    GPUDenseVisualOdometry::GPUDenseVisualOdometry(const int width, const int height)
    {
        width_ = width;
        height_ = height;

        first_frame = true;

        // Camera intrinsics (HARDCODED FOR NOW)
        cam_mat << 211.5657,        0, 210.5820,
                          0, 211.5657, 119.4901,
                          0,        0,        1;
        
        scale = 0.001;

        int gray_size = width*height*sizeof(char), depth_size = width*height*sizeof(short);
        int res_size = width*height*sizeof(float);
        #if (DVO_USE_CUDA > 0)
            std::cout << "Allocating memory" << std::endl;
            HANDLE_CUDA_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
            HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &gray_ptr, gray_size));
            HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &gray_prev_ptr, gray_size));
            HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &depth_prev_ptr, gray_size));
            HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &res_ptr, res_size));
            std::cout << "CUDA Unified memory allocated" << std::endl;
        #else
            gray_ptr = std::malloc(gray_size);
            gray_prev_ptr = std::malloc(gray_size);
            depth_prev_ptr = std::malloc(depth_size);
            res_ptr = std::malloc(res_size);
        #endif
        gray = cv::Mat(height, width, CV_8U, gray_ptr);
        gray_prev = cv::Mat(height, width, CV_8U, gray_ptr);
        depth_prev = cv::Mat(height, width, CV_16U, depth_prev_ptr);
        residuals = cv::Mat(height, width, CV_32F, res_ptr);

        #if (DVO_DEBUG > 0)
            cv::namedWindow("Gray", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("Gray", 0, 0);
            cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("Depth", width_+10, 0);
            cv::namedWindow("Residuals", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("Residuals", 2*(width_+10), 0);
        #endif
    }

    GPUDenseVisualOdometry::~GPUDenseVisualOdometry()
    {
        std::cout << "Exited dtor" << std::endl;
        #if (DVO_USE_CUDA > 0)
            HANDLE_CUDA_ERROR(cudaFree(gray_ptr));
            HANDLE_CUDA_ERROR(cudaFree(gray_prev_ptr));
            HANDLE_CUDA_ERROR(cudaFree(depth_prev_ptr));
            HANDLE_CUDA_ERROR(cudaFree(res_ptr));
        #else
            std::free(gray_ptr);
            std::free(gray_prev_ptr);
            std::free(depth_prev_ptr);
            std::free(res_ptr);
        #endif
    }

    cv::Mat GPUDenseVisualOdometry::step(cv::Mat& color, cv::Mat& depth_in)
    {
        assert(("Image should have the same resolution!", color.cols == depth_in.cols && color.rows == depth_in.rows));

        // Convert color image to grayscale and depth image to 16 bits depth

        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        cv::Mat depth(height_, width_, CV_16U);
        depth_in.convertTo(depth, CV_16U);

        // Initial pose estimation
        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> T = Eigen::Matrix4f::Identity();

        // Check if it is 1st frame
        if(!first_frame)
        {
            // Check for CUDA
            #if(DVO_USE_CUDA > 0)              
                float T_d[4][4], cam_mat_d[3][3];
                std::copy(T.data(), T.data() + 16, *T_d);
                std::copy(cam_mat.data(), cam_mat.data() + 9, *cam_mat_d);

                // Call CUDA Kernel
                call_step_kernel(gray_ptr,
                                 gray_prev_ptr,
                                 depth_prev_ptr,
                                 res_ptr,
                                 T_d,
                                 cam_mat_d,
                                 scale,
                                 width_,
                                 height_);
                                 
            #endif
        }
        else
        {
            first_frame = false;
        }

        // Assign last values
        // gray_last = gray;
        // depth_last = depth;
        gray.copyTo(gray_prev);
        depth.copyTo(depth_prev); 

        #if (DVO_DEBUG > 0)
            // Display images
            cv::imshow("Gray", gray);
            cv::imshow("Depth", depth);
            cv::Mat residuals_scaled;
            cv::convertScaleAbs(residuals, residuals_scaled);
            cv::imshow("Residuals", residuals_scaled);
        #endif

        return residuals;
    }

} // namepsace otto