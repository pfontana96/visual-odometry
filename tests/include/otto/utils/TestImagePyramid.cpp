#include <gtest/gtest.h>

#include <iostream>
#include <cmath>
#include <filesystem>
#include <string>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <eigen3/Eigen/Core>

#include <otto/utils/ImagePyramid.h>

namespace fs = std::filesystem;

void read_testcase(const int index, cv::Mat& gray_image, cv::Mat& depth_image) {
    fs::path cwd = fs::current_path();  // ottobot/build
    fs::path test_data_path = cwd.parent_path() / "tests" / "data";  // ottobot/tests/data

    cv::imread((test_data_path / "rgb" / (std::to_string(index) + ".png")).string(), cv::IMREAD_GRAYSCALE);
    depth_image = cv::imread((test_data_path / "depth" / (std::to_string(index) + ".png")).string(), cv::IMREAD_ANYDEPTH);

    if(gray_image.empty()) 
        throw std::invalid_argument("Could not read gray image at: '" + (test_data_path / "rgb" / (std::to_string(index) + ".png")).string() + "'");

    if(depth_image.empty()) 
        throw std::invalid_argument("Could not read depth image at: '" + (test_data_path / "depth" / (std::to_string(index) + ".png")).string() + "'");
}


TEST(TestImagePyramid, test__given_a_image_and_levels__when_init__then_ok) {
    // Given
    int levels = 5, height = 480, width = 640;
    cv::Mat gray_image(height, width, CV_8U), depth_image(height, width, CV_16U);

    read_testcase(4, gray_image, depth_image);
    std::cout << "gray_image empty: " << gray_image.empty() << std::endl;
    std::cout << "depth_image empty: " << depth_image.empty() << std::endl;

    // When
    otto::RGBDImagePyramid pyr = otto::RGBDImagePyramid(gray_image, depth_image, levels);

    // Then
    std::vector<cv::Mat> gray_pyr = pyr.get_gray_pyramid(), depth_pyr = pyr.get_depth_pyramid();

    std::cout << gray_pyr.size() << std::endl;
    ASSERT_EQ(gray_pyr.size(), levels);
    ASSERT_EQ(depth_pyr.size(), levels);
}