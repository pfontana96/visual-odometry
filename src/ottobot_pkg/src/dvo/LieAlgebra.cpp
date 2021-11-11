#include <LieAlgebra.h>
#include <iostream>
/*
Further info on Lie's Algebra available on: 
    https://ethaneade.com/lie.pdf
    http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf
*/

namespace otto {
    // float limit_angle(float theta)
    // {
    //     theta = fmod(theta + M_PI, 2*M_PI);
    //     if(theta < 0)
    //         theta += 2*M_PI;

    //     return theta - M_PI;
    // }

    // // SO(3)
    // Mat3f SO3_hat(const Eigen::Ref<const Vec3f>& phi)
    // {
    //     Mat3f hat = Mat3f::Zero();
    //     hat(0,1) = -phi(2);
    //     hat(0,2) = phi(1);
    //     hat(1,0) = phi(2);
    //     hat(1,2) = -phi(0);
    //     hat(2,0) = -phi(1);
    //     hat(2,1) = phi(0);

    //     return hat;
    // }

    // Mat3f SO3_exp(const Eigen::Ref<const Vec3f>& phi)
    // {
    //     float theta = phi.squaredNorm();

    //     // Check for singularity at 0
    //     if(abs(theta) < LIE_EPSILON)
    //         return Mat3f::Identity();

    //     Vec3f a = phi/theta;

    //     // Wrap angle between [-pi, pi)
    //     theta = limit_angle(theta);

    //     Mat3f a_hat =  SO3_hat(a);
    //     Mat3f R = Mat3f::Identity() + (sin(theta)/theta)*a_hat + ((1 - cos(theta))/(theta*theta))*a_hat*a_hat;
    //     return R;
    // }

    // // SE(3)
    // Mat4f SE3_hat(const Eigen::Ref<const Vec6f>& xi)
    // {
    //     Mat4f T = Mat4f::Zero();
    //     T.topLeftCorner<3,3>() = SO3_hat(xi.tail<3>());
    //     T.topRightCorner<3,1>() = xi.head<3>();

    //     return T;
    // }

    // Mat4f SE3_exp(const Eigen::Ref<const Vec6f>& xi)
    // {
    //     Mat4f T = Mat4f::Identity();
    //     float theta = xi.tail<3>().squaredNorm();

    //     // Check for singularity at 0
    //     if(abs(theta) < LIE_EPSILON)
    //     {
    //         T.topRightCorner<3,1>() = xi.head<3>();
    //     }else{
    //         float A, B, C;
    //         A = sin(theta)/theta;
    //         B = (1 - cos(theta))/(theta*theta);
    //         C = (1 - A)/(theta*theta);

    //         Mat3f phi_hat = SO3_hat(xi.tail<3>());
    //         Mat3f phi_hat_2 = phi_hat*phi_hat;
    //         Mat3f R = Mat3f::Identity() + A*phi_hat + B*phi_hat_2;
    //         Mat3f V = Mat3f::Identity() + A*phi_hat + C*phi_hat_2;
            
    //         T.topLeftCorner<3,3>() = R;
    //         T.topRightCorner<3,1>() = V*xi.head<3>(); 
    //     }

    //     return T;
    // }

    namespace lie{

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
            float theta = phi.norm();

            // Check for singularity at 0
            if(abs(theta) < LIE_EPSILON)
                return Mat3f::Identity();

            Vec3f a = phi/theta;

            // Wrap angle between [-pi, pi)
            theta = limit_angle(theta);

            Mat3f a_hat =  SO3_hat(a);

            // Rodrigues Formula
            Mat3f R = Mat3f::Identity() + sin(theta)*a_hat + (1.0f - cos(theta))*(a_hat*a_hat);
            return R;
        }

        Vec3f SO3_log(const Eigen::Ref<const Mat3f>& R)
        {
            assert(("Invalid Rotation Matrix", is_rotation_matrix(R)));

            Vec3f phi = Vec3f::Zero();
            float theta = acos((R.trace() - 1.0f)/2.0f);

            if (abs(theta) < LIE_EPSILON)
                return Vec3f::Zero();

            // Wrap angle between [-pi, pi)
            theta = limit_angle(theta);
        
            Mat3f phi_hat = (theta/(2.0f*sin(theta)))*(R - R.transpose());
            phi(0) = phi_hat(2,1);
            phi(1) = phi_hat(0,2);
            phi(2) = phi_hat(1,0);

            return phi;
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
            float theta = xi.tail<3>().norm();
            
            // Wrap angle between [-pi,pi)
            theta = limit_angle(theta);

            // Check for singularity at 0
            if(abs(theta) < LIE_EPSILON)
            {
                T.topRightCorner<3,1>() = xi.head<3>();

            }else{

                Mat4f xi_hat = SE3_hat(xi);
                Mat4f xi_hat_2 = xi_hat*xi_hat; 
                float theta_2 = theta*theta;
                T += xi_hat + ((1 - cos(theta))/theta_2)*xi_hat_2 + ((theta - sin(theta))/(theta_2*theta))*(xi_hat_2*xi_hat);
            }

            return T;
        }

        Vec6f SE3_log(const Eigen::Ref<const Mat4f>& T)
        {
            Vec6f xi;
            xi.tail<3>() = SO3_log(T.topLeftCorner<3,3>());
            float A, B, C, theta;
            theta = xi.tail<3>().norm(); // Angle is already wrapped in [-pi; pi)
            
            // Mat3f phi_hat = SO3_hat(xi.tail<3>());
            Vec3f a = xi.tail<3>()/theta;
            Mat3f a_hat = SO3_hat(a);

            if(abs(theta) < LIE_EPSILON)
            {
                xi.tail<3>() = Vec3f::Zero();
                xi.head<3>() = T.topRightCorner<3,1>();

            }else{    

                float theta_2 = theta/2;
                float A = theta_2*cos(theta_2)/sin(theta_2);
                Mat3f V_inv = A*Mat3f::Identity() + (1 - A)*a*a.transpose() - theta_2*a_hat;

                xi.head<3>() = V_inv*T.topRightCorner<3,1>();
            }
            
            return xi;
        }
    } //namespace lie

} // namespace otto