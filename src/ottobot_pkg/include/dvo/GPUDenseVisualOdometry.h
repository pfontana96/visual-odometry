#ifndef DENSE_VISUAL_ODOMETRY_H
#define DENSE_VISUAL_ODOMETRY_H

#include <eigen3/Eigen/Core>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// ROS
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

// Check for CUDA
#ifdef __CUDACC__
    #include <DenseVisualOdometryKernel.cuh>
#endif

namespace otto
{
    class GPUDenseVisualOdometry
    {
        public:

            GPUDenseVisualOdometry(ros::NodeHandle& nh);
            ~GPUDenseVisualOdometry();

            cv::Mat step(const cv::Mat& color, const cv::Mat& depth);

        private:
            // ROS
            ros::NodeHandle nh_;

            // Attributes
            bool first_frame;

            cv::Mat gray;
            cv::Mat gray_last;
            cv::Mat depth_last;

            // Camera
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> cam_mat;
            float scale;
    }; // class GPUDense Visual Odometry
} // namespace otto
#endif
