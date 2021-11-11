#include <gtest/gtest.h>
#include <iostream>

// dvo
#include <GPUDenseVisualOdometry.h>
#include <LieAlgebra.h>
#include <types.h>

using namespace otto;

// Lie Algebra
TEST(SO3TestSuit, HandlesZeroAngleRotation)
{
  // Zero angle rotations
  ASSERT_TRUE(lie::SO3_exp(Vec3f::Zero()).isApprox(Mat3f::Identity(), LIE_EPSILON));
  ASSERT_TRUE(lie::SO3_log(Mat3f::Identity()).isApprox(Vec3f::Zero(), LIE_EPSILON));
}

TEST(SO3TestSuit, HandlesBackAndForthExample)
{
  // Back and forth conversion
  Vec3f original, inverse;
  original << 1.5708, -0.7854, 0.5236;
  Mat3f R = lie::SO3_exp(original);
  inverse = lie::SO3_log(R);
  ASSERT_TRUE(original.isApprox(inverse, LIE_EPSILON));
}

TEST(SO3TestSuit, HandlesPreDefinedExamples)
{ 
  // Known testcase
  Mat3f R;
  Vec3f phi;

  R << 0.3048809, 0.9520282, -0.0262667,
      -0.8170195, 0.2472735, -0.5208982,
      -0.4894148, 0.1802723, 0.8532146;
  phi << 0.4893, 0.3232, -1.2345;

  ASSERT_TRUE(lie::SO3_exp(phi).isApprox(R, LIE_EPSILON));
  ASSERT_TRUE(lie::SO3_log(R).isApprox(phi, LIE_EPSILON));
}

TEST(SE3TestSuite, HandlesZeroTransform)
{
  ASSERT_TRUE(lie::SE3_exp(Vec6f::Zero()).isApprox(Mat4f::Identity(), LIE_EPSILON));
  ASSERT_TRUE(lie::SE3_log(Mat4f::Identity()).isApprox(Vec6f::Zero(), LIE_EPSILON));
}

TEST(SE3TestSuit, HandlesBackAndForthExample)
{
  // Back and forth conversion
  Vec6f original, inverse;
  original << 0.25, -1.58, 2.987, 1.5708, -0.7854, 0.5236;
  Mat4f T = lie::SE3_exp(original);
  inverse = lie::SE3_log(T);
  ASSERT_TRUE(original.isApprox(inverse, LIE_EPSILON));
}

TEST(SE3TestSuit, HandlesPreDefinedExamples)
{ 
  // Known testcase
  Mat4f T;
  Vec6f xi;

  T << 0.3048809, 0.9520282, -0.0262667, -0.8325922,
      -0.8170195, 0.2472735, -0.5208982, -1.8249098,
      -0.4894148, 0.1802723,  0.8532146, -5.1481801,
               0,         0,          0,      1;

  xi << 1.4852, -3.156, -4.578, 0.4893, 0.3232, -1.2345;

  ASSERT_TRUE(lie::SE3_exp(xi).isApprox(T, LIE_EPSILON));
  ASSERT_TRUE(lie::SE3_log(T).isApprox(xi, LIE_EPSILON));
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}