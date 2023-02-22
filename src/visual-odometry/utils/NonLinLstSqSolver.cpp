#include <utils/NonLinLstSqSolver.h>

namespace vo {
    namespace util {
        NonLinearLeastSquaresSolver::NonLinearLeastSquaresSolver():
            error_(0.0f),
            count_(0)
        {
            A_.setZero();
            b_.setZero();
        }

        NonLinearLeastSquaresSolver::~NonLinearLeastSquaresSolver(){}

        void NonLinearLeastSquaresSolver::update(
            const Eigen::Ref<const Eigen::Matrix<float, 1, 6, Eigen::RowMajor>> J_i,
            const float residual_i, const float weight_i
        ) {

            if (J_i.hasNaN())
                return;

            A_ += (J_i.transpose() * J_i) * weight_i;
            b_ += J_i.transpose() * (-residual_i * weight_i);
            error_ += weight_i * (residual_i * residual_i);
            count_ += 1;
        }

        void NonLinearLeastSquaresSolver::reset() {
            error_ = 0.0f;
            count_ = 0;
            A_.setZero();
            b_.setZero();
        }

        float NonLinearLeastSquaresSolver::solve(vo::util::Vec6f& solution) {
            if (count_ < 6) {
                throw std::invalid_argument(
                    "Not enough equations to solve system, needed at least '6' but got '" +
                    std::to_string(count_) + "'."
                );
            }

            solution = A_.ldlt().solve(b_);

            return (error_ / count_);
        }
    } // namespace util
} // namespace vo