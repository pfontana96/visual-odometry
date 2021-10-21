#ifndef DVO_LIE_ALGEBRA
#define DVO_LIE_ALGEBRA

#include <eigen3/Eigen/Core>
#include <types.h>

#include <math.h>

#include <iostream>

#define LIE_EPSILON 1e-6

namespace otto {
    // Lie's algebra

    // SO(3)
    Mat3f SO3_hat(const Eigen::Ref<const Vec3f>& phi);
    Mat3f SO3_exp(const Eigen::Ref<const Vec3f>& phi);

    // SE(3)
    Mat4f SE3_hat(const Eigen::Ref<const Vec6f>& xi);
    Mat4f SE3_exp(const Eigen::Ref<const Vec6f>& xi);

    // utils
    float limit_angle(float theta);

} // namespace otto

#endif