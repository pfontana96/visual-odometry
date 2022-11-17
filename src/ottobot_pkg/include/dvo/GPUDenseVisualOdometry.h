#ifndef DENSE_VISUAL_ODOMETRY_H
#define DENSE_VISUAL_ODOMETRY_H

#define DVO_DEBUG 1
#define DVO_CHRONO 1

#include <assert.h>
#include <cmath>
#include <iostream>
#include <string.h>
#define __DIRNAME__ (std::string(__FILE__).substr(0, (strrchr(__FILE__, '/') + 1) - __FILE__))

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Cholesky>

#include <manif/manif.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#if (DVO_DEBUG > 0)
    #include <opencv2/highgui.hpp>
#endif

// YAML
#include <yaml-cpp/yaml.h>

// DVO files
#include <DenseVisualOdometryKernel.cuh>
#include <types.h>
#include <LieAlgebra.h>

namespace otto
{
    class GPUDenseVisualOdometry
    {
        public:

            GPUDenseVisualOdometry(const std::string config_file, const int width, const int height);
            // GPUDenseVisualOdometry(std::string config_file);
            ~GPUDenseVisualOdometry();

            Mat4f step(cv::Mat& color, cv::Mat& depth);
            
            Mat3f get_camera_matrix();
            float get_scale();

        private:

            void compute_residuals();
            void weighting();
            void compute_jacobian();
            Mat4f do_gauss_newton();

            int width_, height_;

            /* ATTRIBUTES */
            bool first_frame;

            // Images
            cv::Mat gray, gray_prev, depth_prev, residuals, weights;
            unsigned char *gray_ptr, *gray_prev_ptr;
            unsigned short* depth_prev_ptr; 
            float* res_ptr, *weights_ptr;

            // Camera
            Mat3f cam_mat;
            float scale;
            
            // Camera Pose buffer
            float* T_ptr;

            // Jacobian buffer
            float* J_ptr;

            // Robust weight estimation
            static constexpr float SIGMA_INITIAL = 5.0f;
            static constexpr float DOF_DEFAULT = 5.0f;

            static constexpr int GN_MAX_ITER = 100;
            
    }; // class GPUDense Visual Odometry
} // namespace otto
#endif
