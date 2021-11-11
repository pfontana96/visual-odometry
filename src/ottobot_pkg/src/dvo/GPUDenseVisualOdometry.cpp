#include <GPUDenseVisualOdometry.h>

#if (DVO_CHRONO > 0)
    #include <chrono>
    #include <iomanip>
    using Time = std::chrono::high_resolution_clock;
    using double_sec = std::chrono::duration<double>;
    using time_point = std::chrono::time_point<Time, double_sec>;
#endif

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

        // CUDA Unified Memory Allocation
        int U8C1_size = width*height*sizeof(char), U16C1_size = width*height*sizeof(short);
        int F32C1_size = width*height*sizeof(float);

        HANDLE_CUDA_ERROR(cudaSetDeviceFlags(cudaDeviceMapHost));
        HANDLE_CUDA_ERROR(cudaMemcpyToSymbol(CM, cam_mat.data(), 9*sizeof(float)));
        HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &gray_ptr, U8C1_size));
        HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &gray_prev_ptr, U8C1_size));
        HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &depth_prev_ptr, U16C1_size));
        HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &res_ptr, F32C1_size));
        HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &weights_ptr, F32C1_size));            
        HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &T_ptr, 16*sizeof(float)));
        HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &J_ptr, 6*F32C1_size));
        std::cout << "CUDA Unified memory allocated" << std::endl;

        // Images
        gray = cv::Mat(height, width, CV_8UC1, gray_ptr);
        gray_prev = cv::Mat(height, width, CV_8UC1, gray_prev_ptr);
        depth_prev = cv::Mat(height, width, CV_16UC1, depth_prev_ptr);
        residuals = cv::Mat(height, width, CV_32FC1, res_ptr);
        weights = cv::Mat(height, width, CV_32FC1, weights_ptr);
    
        #if (DVO_DEBUG > 0)
            cv::startWindowThread();
            cv::namedWindow("Gray", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("Gray", 0, 0);
            cv::namedWindow("Gray prev", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("Gray prev", width_+10, 0);
            cv::namedWindow("Residuals", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("Residuals", 2*(width_+10), 0);
            cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("Depth", 0, height_+10);
            cv::namedWindow("Weights", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("Weights", 0, 2*(height_+10));
        #endif
    }

    GPUDenseVisualOdometry::~GPUDenseVisualOdometry()
    {
        std::cout << "Exited dtor" << std::endl;

        HANDLE_CUDA_ERROR(cudaFree(gray_ptr));
        HANDLE_CUDA_ERROR(cudaFree(gray_prev_ptr));
        HANDLE_CUDA_ERROR(cudaFree(depth_prev_ptr));
        HANDLE_CUDA_ERROR(cudaFree(res_ptr));
        HANDLE_CUDA_ERROR(cudaFree(weights_ptr));
        HANDLE_CUDA_ERROR(cudaFree(T_ptr));
        HANDLE_CUDA_ERROR(cudaFree(J_ptr));

    }

    Mat4f GPUDenseVisualOdometry::step(cv::Mat& color, cv::Mat& depth_in)
    {
        #if (DVO_CHRONO > 0)
            time_point start = Time::now();
        #endif
        assert(("Image should have the same resolution!", color.cols == depth_in.cols && color.rows == depth_in.rows));

        // Convert color image to grayscale and depth image to 16 bits depth

        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
        cv::Mat depth(height_, width_, CV_16UC1);
        depth_in.convertTo(depth, CV_16UC1);
        Mat4f T;

        // Check if it is 1st frame
        if(!first_frame)
        {
            T = do_gauss_newton();
        }
        else
        {   
            T = Mat4f::Identity();
            first_frame = false;
        }

        // Assign last values
        gray.copyTo(gray_prev);
        depth.copyTo(depth_prev); 

        #if (DVO_DEBUG > 0)
            // Display images
            cv::imshow("Gray", gray);
            cv::imshow("Gray prev", gray_prev);
            
            double min, max;
            cv::Mat residuals_scaled;
            cv::minMaxIdx(residuals, &min, &max);
            cv::convertScaleAbs(residuals, residuals_scaled, 255/max);
            cv::imshow("Residuals", residuals_scaled);
            
            cv::minMaxIdx(depth, &min, &max);
            cv::Mat depth_cmap, depth_adj;
            cv::convertScaleAbs(depth, depth_adj, 255/max);
            cv::applyColorMap(depth_adj, depth_cmap, cv::COLORMAP_JET);
            cv::imshow("Depth", depth_cmap);

            cv::Mat weights_scaled;
            cv::minMaxIdx(weights, &min, &max);
            cv::convertScaleAbs(weights, weights_scaled, 255/max);
            cv::imshow("Weights", weights_scaled);
        #endif

        #if (DVO_CHRONO > 0)
            time_point end = Time::now();
            std::cout << "Elapsed time: " << (end - start).count() << std::endl;
        #endif

        return T;
    }

    void GPUDenseVisualOdometry::weighting()
    {
        float *var;
        HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &var, sizeof(float)));
        *var = SIGMA_INITIAL*SIGMA_INITIAL;

        call_weighting_kernel(  res_ptr,
                                weights_ptr,
                                var,
                                DOF_DEFAULT,
                                width_,
                                height_);

        HANDLE_CUDA_ERROR(cudaFree(var));
    }

    void GPUDenseVisualOdometry::compute_residuals()
    {           
        call_residuals_kernel(  gray_ptr,
                                gray_prev_ptr,
                                depth_prev_ptr,
                                res_ptr,
                                T_ptr,
                                scale,
                                width_,
                                height_);
    }

    void GPUDenseVisualOdometry::compute_jacobian()
    {
        call_jacobian_kernel(   gray_ptr,
                                depth_prev_ptr,
                                J_ptr, 
                                T_ptr,
                                scale,
                                width_,
                                height_);
    }

    Mat4f GPUDenseVisualOdometry::do_gauss_newton()
    {

        // Initial pose estimation
        Eigen::Map<Mat4f> T(T_ptr);
        T = Mat4f::Identity();

        float error_prev = std::numeric_limits<float>::max();

        float *H_ptr, *b_ptr;
        HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &H_ptr, sizeof(float)*6*6));
        HANDLE_CUDA_ERROR(cudaMallocManaged((void**) &b_ptr, sizeof(float)*6));

        for(int i = 0; i < GN_MAX_ITER; i++)
        {
            compute_residuals();
            
            weighting();

            compute_jacobian();

            float error;
            error = call_newton_gauss_kernel(res_ptr, weights_ptr, J_ptr, H_ptr, b_ptr, width_*height_);

            time_point end_gpu = Time::now();

            Eigen::Map<Eigen::Matrix<float, 6, 6, Eigen::RowMajor>> H(H_ptr, 6, 6);
            Eigen::Map<Vec6f> b(b_ptr, 6, 1);

            Vec6f delta_xi = H.ldlt().solve(b);

            // Update Pose estimation
            T = T*lie::SE3_exp(delta_xi);                  

            // Evaluate convergence
            if(error/error_prev > 0.995)
                break;

            error_prev = error;
        }

        HANDLE_CUDA_ERROR(cudaFree(H_ptr));
        HANDLE_CUDA_ERROR(cudaFree(b_ptr));

        return T;
    }

    Mat3f GPUDenseVisualOdometry::get_camera_matrix()
    {
        return cam_mat;
    }
    
    float GPUDenseVisualOdometry::get_scale()
    {
        return scale;
    }

} // namepsace otto
