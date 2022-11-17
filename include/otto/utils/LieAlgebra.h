#ifndef OTTO_LIE_ALGEBRA
#define OTTO_LIE_ALGEBRA

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>

#include <math.h>

#include <iostream>
#include <cassert>

#include <utils/types.h>

#define LIE_EPSILON 1e-6

namespace otto {
    namespace lie {

        // SO(3)
        namespace SO3 {
            Mat3f hat(const Eigen::Ref<const Vec3f>& phi);
            Mat3f exp(const Eigen::Ref<const Vec3f>& phi);
            Vec3f log(const Eigen::Ref<const Mat3f>& R);
        } // namespace SO3

        // SE(3)
        namespace SE3 {
            Mat4f hat(const Eigen::Ref<const Vec6f>& xi);
            Mat4f exp(const Eigen::Ref<const Vec6f>& xi);
            Vec6f log(const Eigen::Ref<const Mat4f>& T);
            Mat4f inv(const Eigen::Ref<const Mat4f>& T);
        } // namespace SE3

        // utils
        inline float limit_angle(float theta)
        {
            theta = fmod(theta + M_PI, 2*M_PI);
            if(theta < 0)
                theta += 2*M_PI;

            return theta - M_PI;
        }

        inline bool is_rotation_matrix(const Eigen::Ref<const Mat3f>& R) 
        {
            return (((R*R.transpose()).isApprox(Mat3f::Identity(), LIE_EPSILON)) &&
                    ((R.transpose()*R).isApprox(Mat3f::Identity(), LIE_EPSILON)) &&
                    (abs((1.0f - R.determinant())) < LIE_EPSILON));
        }

    } // namespace lie
} // namespace otto

#endif