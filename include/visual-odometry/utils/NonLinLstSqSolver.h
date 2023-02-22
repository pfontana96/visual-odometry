#ifndef VO_UTILS_NONLINEAR_LSTSQ
#define VO_UTILS_NONLINEAR_LSTSQ

#include <exception>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/LU>

#include <utils/types.h>


namespace vo {
    namespace util {

        class NonLinearLeastSquaresSolver {
            public:
                // EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

                // Methods
                NonLinearLeastSquaresSolver();
                ~NonLinearLeastSquaresSolver();

                void update(
                    const Eigen::Ref<const Eigen::Matrix<float, 1, 6, Eigen::RowMajor>> J_i,
                    const float residual_i, const float weight_i = 1.0f
                );

                void reset();
                float solve(vo::util::Vec6f& solution);

            private:
                // Attributes
                int count_;
                float error_;

                vo::util::Mat6f A_;
                vo::util::Vec6f b_;
        };

    } // namspace util
} // namespace vo

#endif
