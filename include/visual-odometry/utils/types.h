#ifndef VO_TYPES_H
#define VO_TYPES_H

#include <limits>

#include <eigen3/Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>

namespace vo {
    namespace util {

        using Mat3f = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
        using Mat4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
        using Mat6f = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>;
        using MatX6f = Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor>;
        using Vec3f = Eigen::Matrix<float, 3, 1>;
        using Vec4f = Eigen::Matrix<float, 4, 1>;
        using Vec6f = Eigen::Matrix<float, 6, 1>;
        using VecXf = Eigen::Matrix<float, Eigen::Dynamic, 1>;

        static const float nan = std::numeric_limits<float>::quiet_NaN();

    } // namespace util
}  // namespace vo

#endif