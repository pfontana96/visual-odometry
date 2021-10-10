#include <ros/ros.h>

// Ottobot libraries
#include <GPUDenseVisualOdometry.h>
#include <DenseVisualOdometryKernel.cuh>

#include <string>

// Debug libraries
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui.hpp>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>

// CUDA
#include <cuda.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "dvo");

    ros::NodeHandle nh;

    otto::GPUDenseVisualOdometry dvo(nh);

    // Count files in dir
    DIR* dp;
    struct dirent *ep;
    std::string imgs_path = "/home/ottobot/Documents/test_data";
    dp = opendir(imgs_path.c_str());

    int nb_imgs = 0;
    if(dp != NULL)
    {
        while (ep = readdir(dp))
            nb_imgs++;
        
        (void) closedir(dp);

        nb_imgs = (int) nb_imgs/2;
        printf("Found %d images", nb_imgs);
    }
    else
        perror("Couldn't open the directory");

    cv::namedWindow("Color");
    cv::namedWindow("Depth");
    
    for(int i=1; i<nb_imgs; i++)
    {
        // Set images filenames
        cv::Mat color = cv::imread(imgs_path + "/color_" + std::to_string(i) + ".png");
        cv::Mat depth = cv::imread(imgs_path + "/depth_" + std::to_string(i) + ".png", 
                                   cv::IMREAD_ANYDEPTH);

        if(color.empty() || depth.empty())
        {
            std::cout << "Couldn't read images" << std::endl;
            continue;
        }

        // Step dvo class
        cv::Mat residuals = dvo.step(color, depth);

        // Show images
        double min, max;
        cv::minMaxIdx(depth, &min, &max);
        cv::Mat depth_cmap, depth_adj;
        cv::convertScaleAbs(depth, depth_adj, 255/max);
        cv::applyColorMap(depth_adj, depth_cmap, cv::COLORMAP_JET);
        
        // Show images
        cv::imshow("Color", color);
        cv::imshow("Depth", depth_cmap);

        // Wait for second frame for residuals
        if(i>1)
            cv::imshow("Residuals", residuals);

        int key = cv::waitKey(1);
        // Check if 'Esc'
        if(key == 27)
        {
            cv::destroyAllWindows();
            break;
        }
    }

    return 0;
}