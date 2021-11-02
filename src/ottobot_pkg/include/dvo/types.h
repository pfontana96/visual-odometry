#ifndef DVO_TYPES_H
#define DVO_TYPES_H

#include <eigen3/Eigen/Core>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

namespace otto {
    using Mat3f = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
    using Mat4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
    using Vec3f = Eigen::Matrix<float, 3, 1>;
    using Vec6f = Eigen::Matrix<float, 6, 1>;
    using RGBPointCloud = pcl::PointCloud<pcl::PointXYZRGB>;
}

#endif
