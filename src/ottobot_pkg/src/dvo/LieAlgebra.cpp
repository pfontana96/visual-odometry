#include <LieAlgebra.h>

namespace otto {
    float limit_angle(float theta)
    {
        theta = fmod(theta + M_PI, 2*M_PI);
        if(theta < 0)
            theta += 2*M_PI;

        return theta - M_PI;
    }

    // SO(3)
    Mat3f SO3_hat(const Eigen::Ref<const Vec3f>& phi)
    {
        Mat3f hat = Mat3f::Zero();
        hat(0,1) = -phi(2);
        hat(0,2) = phi(1);
        hat(1,0) = phi(2);
        hat(1,2) = -phi(0);
        hat(2,0) = -phi(1);
        hat(2,1) = phi(0);

        return hat;
    }

    Mat3f SO3_exp(const Eigen::Ref<const Vec3f>& phi)
    {
        float theta = phi.squaredNorm();

        // Check for singularity at 0
        if(abs(theta) < LIE_EPSILON)
            return Mat3f::Identity();

        Vec3f a = phi/theta;

        // Wrap angle between [-pi, pi)
        theta = limit_angle(theta);

        Mat3f a_hat =  SO3_hat(a);
        Mat3f R = cos(theta)*Mat3f::Identity() + (1 - cos(theta))*(a*a.transpose()) + (1 + sin(theta))*a_hat;
        return R;
    }

    // // SE(3)
    Mat4f SE3_hat(const Eigen::Ref<const Vec6f>& xi)
    {
        Mat4f T = Mat4f::Zero();
        T.topLeftCorner<3,3>() = SO3_hat(xi.tail<3>());
        T.topRightCorner<3,1>() = xi.head<3>();

        return T;
    }

    Mat4f SE3_exp(const Eigen::Ref<const Vec6f>& xi)
    {
        Mat4f T = Mat4f::Identity();
        Mat4f xi_hat = SE3_hat(xi);
        float theta = xi.tail<3>().squaredNorm();

        // Check for singularity at 0
        if(abs(theta) < LIE_EPSILON)
        {
            T.topRightCorner<3,1>() = xi.head<3>();
        }else{
            Mat4f xi_hat_2 = xi_hat*xi_hat;
            T += xi_hat + ((1 - cos(theta))/(pow(theta, 2)))*xi_hat_2 + ((theta - sin(theta))/(pow(theta, 3)))*(xi_hat_2*xi_hat);
        }

        return T;
    }

} // namespace otto