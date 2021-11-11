#ifndef DVO_LIE_ALGEBRA
#define DVO_LIE_ALGEBRA

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <types.h>

#include <math.h>

#include <iostream>
#include <cassert>

#define LIE_EPSILON 1e-6

namespace otto {
    namespace lie{

        // SO(3)
        Mat3f SO3_hat(const Eigen::Ref<const Vec3f>& phi);
        Mat3f SO3_exp(const Eigen::Ref<const Vec3f>& phi);
        Vec3f SO3_log(const Eigen::Ref<const Mat3f>& R);

        // SE(3)
        Mat4f SE3_hat(const Eigen::Ref<const Vec6f>& xi);
        Mat4f SE3_exp(const Eigen::Ref<const Vec6f>& xi);
        Vec6f SE3_log(const Eigen::Ref<const Mat4f>& T);

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

    // // Lie's algebra

    // // SO(3)
    // Mat3f SO3_hat(const Eigen::Ref<const Vec3f>& phi);
    // Mat3f SO3_exp(const Eigen::Ref<const Vec3f>& phi);

    // // SE(3)
    // Mat4f SE3_hat(const Eigen::Ref<const Vec6f>& xi);
    // Mat4f SE3_exp(const Eigen::Ref<const Vec6f>& xi);

    // // utils
    // float limit_angle(float theta);

} // namespace otto

#endif