#include <LieAlgebra.h>
/*
Further info available on: https://ethaneade.com/lie.pdf
*/

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
        Mat3f R = Mat3f::Identity() + (sin(theta)/theta)*a_hat + ((1 - cos(theta))/(theta*theta))*a_hat*a_hat;
        return R;
    }

    // SE(3)
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
        float theta = xi.tail<3>().squaredNorm();

        // Check for singularity at 0
        if(abs(theta) < LIE_EPSILON)
        {
            T.topRightCorner<3,1>() = xi.head<3>();
        }else{
            float A, B, C;
            A = sin(theta)/theta;
            B = (1 - cos(theta))/(theta*theta);
            C = (1 - A)/(theta*theta);

            Mat3f phi_hat = SO3_hat(xi.tail<3>());
            Mat3f phi_hat_2 = phi_hat*phi_hat;
            Mat3f R = Mat3f::Identity() + A*phi_hat + B*phi_hat_2;
            Mat3f V = Mat3f::Identity() + A*phi_hat + C*phi_hat_2;
            
            T.topLeftCorner<3,3>() = R;
            T.topRightCorner<3,1>() = V*xi.head<3>(); 
        }

        return T;
    }

} // namespace otto