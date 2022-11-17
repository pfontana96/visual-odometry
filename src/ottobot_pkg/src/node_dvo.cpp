#include <ros/ros.h>

// Ottobot libraries
#include <GPUDenseVisualOdometry.h>
#include <DenseVisualOdometryKernel.cuh>

#include <DVOROSNode.h>

#include <string>
#include <typeinfo>

// Debug libraries
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

// CUDA
#include <cuda.h>

namespace otto {

    DVOROSNode::DVOROSNode(ros::NodeHandle& nh):
        sync(SyncPolicy(10), color_sub, depth_sub),
        dvo(__DIRNAME__ + "config/camera_intrinsics.yaml", 424, 240),
        tf_listener(tf_buffer),
        rate(15)
    {
        nh_ = nh;

        std::string ns = ros::this_node::getNamespace();

        color_sub.subscribe(nh_, "/ottobot/realsense_camera/raw/rgb_image", 1);
        depth_sub.subscribe(nh_, "/ottobot/realsense_camera/raw/depth_image", 1);

        sync.registerCallback(&DVOROSNode::callback, this);

        pointcloud = nh_.param<bool>("pointcloud", false);

        if(pointcloud)
        {
            ROS_INFO("PointCloud creation enabled");

            cloud.reset(new RGBPointCloud);
            cloud->header.frame_id = "/camera_link";
            cloud->is_dense = false;
            cloud->height = 240;
            cloud->width = 424;
            cloud->points.resize(cloud->height*cloud->width);

            pub_cloud = nh_.advertise<RGBPointCloud>(   ros::this_node::getNamespace()      \
                                                        + "/" + ros::this_node::getName()   \
                                                        + "/cloud", 5);
        }

        // At start we assume robot's at map's origin
        accumulated_transform.setIdentity();
        stamp = ros::Time::now();
    }

    DVOROSNode::~DVOROSNode()
    { 
    }

    void DVOROSNode::callback(  const sensor_msgs::ImageConstPtr& color, 
                                const sensor_msgs::ImageConstPtr& depth)
    {
        cv_bridge::CvImagePtr color_ptr, depth_ptr;
        try
        {
            color_ptr = cv_bridge::toCvCopy(color, sensor_msgs::image_encodings::TYPE_8UC3);
            depth_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_16UC1);
        }
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("%s", e.what());
            return;
        }

        geometry_msgs::TransformStamped tf_msg;
        Eigen::Affine3d base_to_camera;
        // tf2::Stamped<tf2::Transform> cam2base;
        // tf2::Transform base2cam;
        try
        {
            // Get last available transform from base_link to camera_link (static transform)
            tf_msg = tf_buffer.lookupTransform("camera_link", "base_link", color->header.stamp);
            // tf2::fromMsg(tf_msg, cam2base);

            // What we actually want it's the inverse of this transformation
            // base2cam = cam2base.inverse();

            base_to_camera = tf2::transformToEigen(tf_msg);
        }
        catch(tf2::TransformException& e)
        {
            ROS_ERROR("%s", e.what());
            return;
        }

        Mat4f T_dvo = dvo.step(color_ptr->image, depth_ptr->image);
        Eigen::Affine3d camera_odometry;
        camera_odometry.matrix() = T_dvo.cast<double>();

        // Update base_link pose
        accumulated_transform = accumulated_transform*(base_to_camera*camera_odometry);

        stamp = color->header.stamp;

        // Update pointcloud if required
        if(pointcloud)
            create_pointcloud(color_ptr->image, depth_ptr->image);
        
    }

    void DVOROSNode::create_pointcloud( const cv::Mat& color, 
                                        const cv::Mat& depth)
    {

        Mat3f K = dvo.get_camera_matrix();
        float scale = dvo.get_scale();
        float fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);

        int idx = 0;

        const float bad_point = std::numeric_limits<float>::quiet_NaN();

        #pragma omp parallel for schedule(dynamic) num_threads(std::thread::hardware_concurrency())
        for(int y = 0; y < depth.rows; y++)
        {
            const cv::Vec3b* rgb_ptr;
            const ushort* depth_ptr;

            #pragma omp critical
            {
                rgb_ptr = color.ptr<cv::Vec3b>(y);
                depth_ptr = depth.ptr<ushort>(y);
            }

            for(int x = 0; x < depth.cols; x++)
            {
                pcl::PointXYZRGB p;                
                float z = ((float) depth_ptr[x]) * scale;

                if ( z > 0.0f)
                {
                    p.z = z;
                    p.x = p.z*(x - cx)/fx;
                    p.y = p.z*(y - cy)/fy;

                    int b = (int) rgb_ptr[x][0];
                    int g = (int) rgb_ptr[x][1];
                    int r = (int) rgb_ptr[x][2];
                    int rgb = (r << 16) + (g << 8) + b;
                    p.rgb = (float) rgb;
                }
                else
                {
                    p.x = p.y = p.z = p.rgb = bad_point;
                }

                #pragma omp critical
                cloud->points[idx] = p;

                idx++;
            }
        }
    }

    void DVOROSNode::publish_all()
    {
        // Publish pointcloud if required
        if(pointcloud && (pub_cloud.getNumSubscribers() > 0))
            pub_cloud.publish(*cloud);

        // Broadcast robot's pose
        Eigen::Vector3d t = accumulated_transform.translation();
        Eigen::Quaterniond q(accumulated_transform.rotation());
        q.normalize();

        geometry_msgs::TransformStamped tf_msg_out;
        tf_msg_out.header.stamp = stamp;
        tf_msg_out.header.frame_id = "map";
        tf_msg_out.child_frame_id = "base_link";
        tf_msg_out.transform.translation.x = t.x();
        tf_msg_out.transform.translation.y = t.y();
        tf_msg_out.transform.translation.z = t.z();
        tf_msg_out.transform.rotation.x = q.x();
        tf_msg_out.transform.rotation.y = q.y();
        tf_msg_out.transform.rotation.z = q.z();
        tf_msg_out.transform.rotation.w = q.w();

        br.sendTransform(tf_msg_out);
    }

    void DVOROSNode::run()
    {
        while(ros::ok())
        {
            try{
                ros::spinOnce();
                publish_all();
                rate.sleep();
            }
            catch(std::runtime_error& e)
            {
                ROS_ERROR("%s", e.what());
            }
        }
    }

} // namespace otto

int main(int argc, char** argv)
{
    query_devices();
    ros::init(argc, argv, "dvo");

    ros::NodeHandle nh("~");

    otto::DVOROSNode node(nh);

    // ros::spin();
    node.run();

    return 0;

    // otto::GPUDenseVisualOdometry dvo(424, 240);

    // // Count files in dir
    // DIR* dp;
    // struct dirent *ep;
    // std::string imgs_path = "/home/ottobot/Documents/test_data";
    // dp = opendir(imgs_path.c_str());

    // int nb_imgs = 0;
    // if(dp != NULL)
    // {
    //     while (ep = readdir(dp))
    //         nb_imgs++;
        
    //     (void) closedir(dp);

    //     nb_imgs = (int) nb_imgs/2;
    //     printf("Found %d images\n", nb_imgs);
    // }
    // else
    //     perror("Couldn't open the directory");

    // // cv::namedWindow("Color");
    // // cv::namedWindow("Depth");
    // // cv::namedWindow("Residuals");
    
    // for(int i=1; i<nb_imgs; i++)
    // {
    //     // Set images filenames
    //     cv::Mat color = cv::imread(imgs_path + "/color_" + std::to_string(i) + ".png");
    //     cv::Mat depth = cv::imread(imgs_path + "/depth_" + std::to_string(i) + ".png", 
    //                                cv::IMREAD_ANYDEPTH);

    //     if(color.empty() || depth.empty())
    //     {
    //         std::cout << "Couldn't read images" << std::endl;
    //         continue;
    //     }

    //     // Step dvo class
    //     cv::Mat residuals = dvo.step(color, depth);

    //     // Show images
    //     double min, max;
    //     cv::minMaxIdx(depth, &min, &max);
    //     cv::Mat depth_cmap, depth_adj;
    //     cv::convertScaleAbs(depth, depth_adj, 255/max);
    //     cv::applyColorMap(depth_adj, depth_cmap, cv::COLORMAP_JET);

    //     int key = cv::waitKey(1);
    //     // Check if 'Esc'
    //     if(key == 27)
    //     {
    //         cv::destroyAllWindows();
    //         break;
    //     }

    //     if(!ros::ok())
    //         break;
    // }

    // return 0;
}