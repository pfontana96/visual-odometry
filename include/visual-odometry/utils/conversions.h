#ifndef VO_UTILS_CONVERSIONS_H
#define VO_UTILS_CONVERSIONS_H

#include <eigen3/Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <utils/types.h>

// Transformation representation conversions exposed to the Python API
namespace vo {
    namespace util {

        /**
         * @brief Returns quaternion from Rotation matrix
         * 
         * @param rotmat Rotation Matrix
         * @return vo::util::Vec4f Quaternion in format w, x, y, z
         */
        vo::util::Vec4f rotmat_to_quaternion(Eigen::Ref<const vo::util::Mat3f> rotmat) {
            Sophus::SO3f so3(rotmat);
            Eigen::Quaternion<float> quat = so3.unit_quaternion();
            vo::util::Vec4f result;
            result << quat.w(), quat.x(), quat.y(), quat.z();
            return result;
        };

        /**
         * @brief Returns Rotation matrix from quaterion
         * 
         * @param quaternion Quaternion in format w, x, y, z
         * @return vo::util::Mat3f Rotation matrix
         */
        vo::util::Mat3f quaternion_to_rotmat(Eigen::Ref<const vo::util::Vec4f> quaternion) {
            Eigen::Quaternion<float> quat(quaternion(0), quaternion(1), quaternion(2), quaternion(3));
            Sophus::SO3f so3(quat);
            return so3.matrix();
        };

        /**
         * @brief Inverts a Rigid Body transformation
         * 
         * @param T Transformation matrix
         * @return vo::util::Mat4f 
         */
        vo::util::Mat4f inverse(Eigen::Ref<const vo::util::Mat4f> T) {
            Sophus::SE3f se3(T);
            return se3.inverse().matrix();
        };

    } // namespace util
} // namespace vo

#endif