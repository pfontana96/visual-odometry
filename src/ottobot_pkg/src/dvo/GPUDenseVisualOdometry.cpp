#include <GPUDenseVisualOdometry.h>

namespace otto{
    GPUDenseVisualOdometry::GPUDenseVisualOdometry(ros::NodeHandle& nh)
    {
        nh_ = nh;

        first_frame = true;

        // Camera intrinsics (HARDCODED FOR NOW)
        cam_mat << 211.5657,        0, 210.5820,
                          0, 211.5657, 119.4901,
                          0,        0,        1;
        
        scale = 0.001;
    }

    GPUDenseVisualOdometry::~GPUDenseVisualOdometry()
    {

    }

    cv::Mat GPUDenseVisualOdometry::step(const cv::Mat& color, const cv::Mat& depth)
    {
        // Convert color image to grayscale
        cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

        int width = gray.cols, height = gray.rows;
        cv::Mat residuals(height, width, CV_32F);

        // Initial pose estimation
        Eigen::Matrix<float, 4, 4, Eigen::RowMajor> T = Eigen::Matrix4f::Identity();

        // Check if it is 1st frame
        if(!first_frame)
        {
            // Check for CUDA
            #ifdef __CUDACC__              

                // Get device pointers from host memory. No allocation or memcpy
                // Valid because on Nano devices CPU and GPU share physical memory
                unsigned char* gray_d_ptr, gray_last_d_ptr, depth_last_d_ptr;
                cudaHostGetDevicePointer((void**) &gray_d_ptr, (void*) gray.data, 0);
                cudaHostGetDevicePointer((void**) &gray_last_d_ptr, (void*) gray_last.data, 0);
                cudaHostGetDevicePointer((void**) &depth_last_d_ptr, (void*) depth_last.data, 0);

                float* residuals_d_ptr;
                cudaHostGetDevicePointer((void**) &residuals_d_ptr, (void*) residuals.data, 0);

                float T_d[4][4], cam_mat_d[3][3];
                std::copy(T.data(), T.data() + 16, *T_d);
                std::copy(cam_mat.data(), cam_mat.data() + 9, *cam_mat_d)

                // Calculate Nb of threads per block and blocks per grid
                int dx = (int) width/CUDA_BLOCK_SIZE;
                int mx = width%CUDA_BLOCK_SIZE;
                int dy = (int) height/CUDA_BLOCK_SIZE;
                int my = height%CUDA_BLOCK_SIZE;

                dx = mx > 0 ? dx+1, dx;
                dy = my > 0 ? dy+1, dy;
                dim3 blockdim(CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE), griddim(dx, dy);

                // Call Residuals Kernel
                step_kernel<<griddim, blockdim>>(gray_d_ptr,
                                                 gray_last_d_ptr,
                                                 depth_last_d_ptr,
                                                 residuals_d_ptr,
                                                 T_d,
                                                 cam_mat_d,
                                                 scale,
                                                 width,
                                                 height);



            #endif
        }

        // Assign last values
        gray_last = gray;
        depth_last = depth;

        if(first_frame)
            first_frame = false;

        return residuals;
    }

} // namepsace otto