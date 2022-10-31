#include <gtest/gtest.h>

#include <iostream>
#include <cmath>

#include <eigen3/Eigen/Core>

#include <otto/utils/LieAlgebra.h>
#include <otto/utils/types.h>

// SO3
TEST(TestSO3, test__given_phi__when_hat__then_ok) {
  // Given
  otto::Vec3f phi;
  phi << 1, 2, 3;

  // When
  otto::Mat3f phi_hat = otto::lie::SO3::hat(phi);

  // Then
  otto::Mat3f expected_phi_hat;
  expected_phi_hat << 0.0, -3.0, 2.0,
                      3.0, 0.0, -1.0,
                      -2.0, 1.0, 0.0;

  ASSERT_TRUE(phi_hat.isApprox(expected_phi_hat, 1e-6));
}

TEST(TestSO3, test__given_phi__when_exp__then_ok) {
  // Given
  otto::Vec3f phi;
  phi << 30.0, 60.0, -5.0;
  phi = phi * (M_PI / 180.0);

  // When
  otto::Mat3f R = otto::lie::SO3::exp(phi);

  // Then
  otto::Mat3f expected_R;
  expected_R << 0.50845740, 0.3126321,  0.8023292,
                0.17552060, 0.8745719, -0.4520139,
                -0.8430086, 0.3706551,  0.3898093;

  // std::cout << "R\n" << R << std::endl;
  // std::cout << "Expected R\n" << expected_R << std::endl;

  ASSERT_TRUE(R.isApprox(expected_R, 1e-6));
}

TEST(TestSO3, test__given_singular_phi__when_exp__then_identity) {
  // Given
  otto::Vec3f phi = otto::Vec3f::Zero();

  // When
  otto::Mat3f R = otto::lie::SO3::exp(phi);

  // Then
  otto::Mat3f expected_R = otto::Mat3f::Identity();

  // std::cout << "R\n" << R << std::endl;
  // std::cout << "Expected R\n" << expected_R << std::endl;

  ASSERT_TRUE(R.isApprox(expected_R, 1e-6));
}

TEST(TestSO3, test__given_rot_mat__when_log__then_ok) {
  // Given
  otto::Mat3f R;
  R << 0.50845740, 0.3126321,  0.8023292,
       0.17552060, 0.8745719, -0.4520139,
       -0.8430086, 0.3706551,  0.3898093;
  
  // When
  otto::Vec3f phi = otto::lie::SO3::log(R);

  // Then
  otto::Vec3f expected_phi;
  expected_phi << 30.0, 60.0, -5.0;
  expected_phi = expected_phi * (M_PI / 180.0);

  // std::cout << "phi\n" << phi << std::endl;
  // std::cout << "Expected phi\n" << expected_phi << std::endl;

  ASSERT_TRUE(phi.isApprox(expected_phi, 1e-6));
}

TEST(TestSO3, test__given_identity_rot_mat__when_log__then_zeros) {
  // Given
  otto::Mat3f R = otto::Mat3f::Identity();
  
  // When
  otto::Vec3f phi = otto::lie::SO3::log(R);

  // Then
  otto::Vec3f expected_phi = otto::Vec3f::Zero();

  // std::cout << "phi\n" << phi << std::endl;
  // std::cout << "Expected phi\n" << expected_phi << std::endl;

  ASSERT_TRUE(phi.isApprox(expected_phi, 1e-6));
}


// SE3
TEST(TestSE3, test__given_xi__when_hat__then_ok) {
  // Given
  otto::Vec6f xi;
  xi << 1, 2, 3, 4, 5, 6;

  // When
  otto::Mat4f xi_hat = otto::lie::SE3::hat(xi);

  // Then
  otto::Mat4f expected_xi_hat;
  expected_xi_hat << 0.0, -6.0, 5.0, 1.0,
                     6.0, 0.0, -4.0, 2.0,
                     -5.0, 4.0, 0.0, 3.0,
                     0.0, 0.0, 0.0, 0.0;

  ASSERT_TRUE(xi_hat.isApprox(expected_xi_hat, 1e-6));
}

TEST(TestSE3, test__given_xi__when_exp__then_ok) {
  // Given
  otto::Vec6f xi;
  xi << 1.4852, -3.156, -4.578, 0.4893, 0.3232, -1.2345;

  // When
  otto::Mat4f T = otto::lie::SE3::exp(xi);

  // Then
  otto::Mat4f expected_T;
  expected_T << 0.30488090, 0.9520282, -0.0262667, -0.8325922,
                -0.8170195, 0.2472735, -0.5208982, -1.8249098,
                -0.4894148, 0.1802723,  0.8532146, -5.1481801,
                0.00000000, 0.0000000,  0.0000000, 1.00000000;

  // std::cout << "T\n" << T << std::endl;
  // std::cout << "Expected T\n" << expected_T << std::endl;

  ASSERT_TRUE(T.isApprox(expected_T, 1e-6));
}

TEST(TestSE3, test__given_singular_xi__when_exp__then_identity) {
  // Given
  otto::Vec6f xi = otto::Vec6f::Zero();

  // When
  otto::Mat4f T = otto::lie::SE3::exp(xi);

  // Then
  otto::Mat4f expected_T = otto::Mat4f::Identity();

  // std::cout << "T\n" << T << std::endl;
  // std::cout << "Expected T\n" << expected_T << std::endl;

  ASSERT_TRUE(T.isApprox(expected_T, 1e-6));
}

TEST(TestSE3, test__given_T__when_log__then_ok) {
  // Given
  otto::Mat4f T;
  T << 0.30488090, 0.9520282, -0.0262667, -0.8325922,
       -0.8170195, 0.2472735, -0.5208982, -1.8249098,
       -0.4894148, 0.1802723,  0.8532146, -5.1481801,
       0.00000000, 0.0000000,  0.0000000, 1.00000000;
  
  // When
  otto::Vec6f xi = otto::lie::SE3::log(T);

  // Then
  otto::Vec6f expected_xi;
  expected_xi << 1.4852, -3.156, -4.578, 0.4893, 0.3232, -1.2345;

  // std::cout << "xi\n" << xi << std::endl;
  // std::cout << "Expected xi\n" << expected_xi << std::endl;

  ASSERT_TRUE(xi.isApprox(expected_xi, 1e-6));
}

TEST(TestSE3, test__given_identity_T__when_log__then_zeros) {
  // Given
  otto::Mat4f T = otto::Mat4f::Identity();
  
  // When
  otto::Vec6f xi = otto::lie::SE3::log(T);

  // Then
  otto::Vec6f expected_xi = otto::Vec6f::Zero();

  // std::cout << "xi\n" << xi << std::endl;
  // std::cout << "Expected xi\n" << expected_xi << std::endl;

  ASSERT_TRUE(xi.isApprox(expected_xi, 1e-6));
}
