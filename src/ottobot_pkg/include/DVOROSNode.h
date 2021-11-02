#ifndef DVO_NODE_H
#define DVO_NODE_H

// Ottobot Library
#include <dvo/GPUDenseVisualOdometry.h>

// ROS libraries
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Transform.h>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <geometry_msgs/TransformStamped.h>

// PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl_ros/point_cloud.h>

// Standard
#include <limits>
#include <thread>

#include <omp.h>

namespace otto{
    class DVOROSNode {

        using Image = sensor_msgs::Image;
        using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;
        using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

        public:

            void callback(  const sensor_msgs::ImageConstPtr& color, 
                            const sensor_msgs::ImageConstPtr& depth);

            DVOROSNode(ros::NodeHandle& nh);
            ~DVOROSNode();

        private:

            void create_pointcloud( const cv::Mat& color, 
                                    const cv::Mat& depth, 
                                    RGBPointCloud::Ptr& cloud);

            void publish_pointcloud(RGBPointCloud::Ptr& cloud);

            ros::NodeHandle nh_;

            bool pointcloud;

            message_filters::Subscriber<Image> color_sub, depth_sub;
            Synchronizer sync;

            GPUDenseVisualOdometry dvo;

            tf2_ros::TransformBroadcaster br;
            tf2_ros::Buffer tf_buffer;
            tf2_ros::TransformListener tf_listener;

            ros::Publisher pub_cloud;

    }; // class DVONode
} // namespace otto

#endif
