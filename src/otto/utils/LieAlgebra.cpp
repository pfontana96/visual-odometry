/*
Further info on Lie's Algebra available on: 
    http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf
*/
#include <iostream>

#include <utils/LieAlgebra.h>


namespace otto {

    namespace lie{
        // SO(3)
        namespace SO3 {
            Mat3f hat(const Eigen::Ref<const Vec3f>& phi) {
                Mat3f hat = Mat3f::Zero();
                hat(0,1) = -phi(2);
                hat(0,2) = phi(1);
                hat(1,0) = phi(2);
                hat(1,2) = -phi(0);
                hat(2,0) = -phi(1);
                hat(2,1) = phi(0);

                return hat;
            }

            Mat3f exp(const Eigen::Ref<const Vec3f>& phi) {
                float theta = phi.norm();

                // Check for singularity at 0
                if(abs(theta) < LIE_EPSILON)
                    return Mat3f::Identity();

                Vec3f a = phi / theta;

                // Wrap angle between [-pi, pi)
                theta = limit_angle(theta);
                Mat3f a_hat =  SO3::hat(a);

                // Rodrigues Formula
                Mat3f R = Mat3f::Identity() + sin(theta) * a_hat + (1.0f - cos(theta))*(a_hat * a_hat);
                return R;
            }

            Vec3f log(const Eigen::Ref<const Mat3f>& R) {
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
        }

        // SE(3)
        namespace SE3 {
            Mat4f hat(const Eigen::Ref<const Vec6f>& xi) {
                Mat4f T = Mat4f::Zero();
                T.topLeftCorner<3,3>() = SO3::hat(xi.tail<3>());
                T.topRightCorner<3,1>() = xi.head<3>();

                return T;
            }

            Mat4f exp(const Eigen::Ref<const Vec6f>& xi) {

                float theta = xi.tail<3>().norm();
                
                // Wrap angle between [-pi,pi)
                theta = limit_angle(theta);

                Mat4f T = Mat4f::Identity();

                // Check for singularity at 0
                if(abs(theta) < LIE_EPSILON)
                {
                    T.topRightCorner<3,1>() = xi.head<3>();

                }else{

                    Mat4f xi_hat = hat(xi);
                    Mat4f xi_hat_2 = xi_hat*xi_hat; 
                    float theta_2 = theta*theta;
                    T += xi_hat + ((1 - cos(theta))/theta_2)*xi_hat_2 + ((theta - sin(theta))/(theta_2*theta))*(xi_hat_2*xi_hat);
                }

                return T;
            }

            Vec6f log(const Eigen::Ref<const Mat4f>& T) {

                Vec6f xi;
                xi.tail<3>() = SO3::log(T.topLeftCorner<3,3>());
                
                float theta;
                theta = xi.tail<3>().norm(); // Angle is already wrapped in [-pi; pi)

                if(abs(theta) < LIE_EPSILON)
                {
                    xi.tail<3>() = Vec3f::Zero();
                    xi.head<3>() = T.topRightCorner<3,1>();

                }else{    

                    Vec3f a = xi.tail<3>() / theta;
                    Mat3f a_hat = SO3::hat(a);

                    float theta_2 = theta / 2;
                    float A = theta_2 * (cos(theta_2) / sin(theta_2));
                    Mat3f V_inv = (A * Mat3f::Identity()) + (1 - A) * (a * a.transpose()) - (theta_2 * a_hat);

                    xi.head<3>() = V_inv*T.topRightCorner<3,1>();
                }
                
                return xi;
            }

            Mat4f inv(const Eigen::Ref<const Mat4f>& T) {
                Mat4f T_inverse = Mat4f::Identity();
                T_inverse.topLeftCorner<3,3>() = T.topLeftCorner<3,3>().transpose();
                T_inverse.topRightCorner<3,1>() = - T.topLeftCorner<3,3>().transpose() * T.topRightCorner<3,1>();

                return T_inverse;
            }
        }

    } //namespace lie

} // namespace otto