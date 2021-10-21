#ifndef DENSE_VISUAL_ODOMETRY_H
#define DENSE_VISUAL_ODOMETRY_H

#define DVO_USE_CUDA 1
#define DVO_DEBUG 0

#include <assert.h>
#include <cmath>
#include <iostream>

#include <eigen3/Eigen/Core>

#include <manif/manif.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#ifdef DVO_DEBUG
    #include <opencv2/highgui.hpp>
#endif

// DVO files
#ifdef DVO_USE_CUDA
    #include <DenseVisualOdometryKernel.cuh>
#endif
#include <types.h>
#include <LieAlgebra.h>

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

            /* ATTRIBUTES */
            bool first_frame;

            // Images
            cv::Mat gray, gray_prev, depth_prev, residuals, weights;
            unsigned char *gray_ptr, *gray_prev_ptr;
            unsigned short* depth_prev_ptr; 
            float* res_ptr, *weights_ptr;
            
            // Camera
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> cam_mat;
            float scale;

            // Camera Pose buffer
            float* T_ptr;

            // Jacobian buffer
            float* J_ptr;

            // Robust weight estimation
            static constexpr float SIGMA_INITIAL = 5.0f;
            static constexpr float DOF_DEFAULT = 5.0f;

            static constexpr int GN_MAX_ITER = 100;

            /* FUNCTIONS */
            void compute_residuals();
            void weighting();
            void compute_jacobian();
            void do_gauss_newton();
            
    }; // class GPUDense Visual Odometry
} // namespace otto
#endif
