#ifndef DENSE_VISUAL_ODOMETRY_H
#define DENSE_VISUAL_ODOMETRY_H

#define DVO_USE_CUDA 1
#define DVO_DEBUG 1

#include <assert.h>
#include <cmath>
#include <iostream>

#include <eigen3/Eigen/Core>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#ifdef DVO_DEBUG
    #include <opencv2/highgui.hpp>
#endif

#ifdef DVO_USE_CUDA
    #include <DenseVisualOdometryKernel.cuh>
#endif

namespace otto
{
    class GPUDenseVisualOdometry
    {
        public:

            GPUDenseVisualOdometry(const int width, const int height);
            ~GPUDenseVisualOdometry();

            cv::Mat step(cv::Mat& color, cv::Mat& depth);

        private:
            int width_, height_;

            // Attributes
            bool first_frame;

            // Images
            cv::Mat gray, gray_prev, depth_prev, residuals;
            unsigned char *gray_ptr, *gray_prev_ptr;
            unsigned short* depth_prev_ptr; 
            float* res_ptr;
            
            // Camera
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> cam_mat;
            float scale;
    }; // class GPUDense Visual Odometry
} // namespace otto
#endif
