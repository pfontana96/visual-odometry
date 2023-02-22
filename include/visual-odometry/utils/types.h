#ifndef VO_TYPES_H
#define VO_TYPES_H

#include <eigen3/Eigen/Core>

// #include <pcl/point_types.h>
// #include <pcl/point_cloud.h>

namespace vo {
    namespace util {
        using Mat3f = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
        using Mat4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
        using Mat6f = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>;
        using MatX6f = Eigen::Matrix<float, Eigen::Dynamic, 6, Eigen::RowMajor>;
        using Vec3f = Eigen::Matrix<float, 3, 1>;
        using Vec6f = Eigen::Matrix<float, 6, 1>;

        template<typename T>
        class Revertable {
            public:
                Revertable(const T& value) : value(value)
                {
                }

                inline const T& operator()() const
                {
                    return value;
                }

                T& update()
                {
                    old = value;

                    return value;
                }

                void revert()
                {
                    value = old;
                }
            private:
                T old, value;
        };

    } // namespace util
}  // namespace vo

#endif