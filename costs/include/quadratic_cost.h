#ifndef TORC_QUADRATIC_COST_H
#define TORC_QUADRATIC_COST_H

#include <string>
#include "base_cost.h"
#include "linear_cost.h"

namespace torc::cost {
    /**
     * Class implementation of a linear cost function, f(x) = x^T A x, where A is a symmetric matrix.
     * @tparam scalar_t the type of scalar used for the cost
     */
    template <class scalar_t>
    class QuadraticCost: public BaseCost<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;

    public:
        explicit QuadraticCost(const matrixx_t& coefficients, const std::string& identifier="Quadratic cost") {
            if ((coefficients.transpose() - coefficients).squaredNorm() != 0) {
                throw std::runtime_error("Quadratic cost: matrix must be symmetric.");
            }
            this->A_ = coefficients;
            this->linear_cost_ = LinearCost(vectorx_t(vectorx_t::Zero(coefficients.cols())));
            this->identifier_ = identifier;
            this->domain_dim_ = coefficients.cols();
        }

        explicit QuadraticCost(const matrixx_t& coefficients, const LinearCost<scalar_t>& linear_cost,
                               const std::string& identifier="Quadratic cost") :
               QuadraticCost(coefficients, identifier) { this->linear_cost_ = linear_cost; }

        explicit QuadraticCost(const matrixx_t& coefficients, const vectorx_t& lin_coefficients,
                               const std::string& identifier="Quadratic cost", const std::string& lin_identifier="linear cost") :
                QuadraticCost(coefficients, identifier) { this->linear_cost_ = LinearCost<scalar_t>(lin_coefficients, lin_identifier); }

        template <int dim>
        explicit QuadraticCost(const Eigen::TriangularView<Eigen::Matrix<scalar_t, dim, dim>, Eigen::Upper>& coefficients,
                               const std::string& identifier="Quadratic cost") :
               QuadraticCost(matrixx_t(matrixx_t(coefficients).template selfadjointView<Eigen::Upper>()), identifier) {}

        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return (1/2) x^T A x + q^T x
         */
        scalar_t Evaluate(const vectorx_t& x) const {
            return x.dot(A_ * x) * 0.5 + linear_cost_.Evaluate(x);
        }

        /**
         * Get the A of the cost.
         * @return the A_
         */
        matrixx_t GetQuadCoefficients() const {
            return A_;
        }

        /**
         * Get the q of the cost
         * @return the q_
         */
        vectorx_t GetLinCoefficients() const {
            return linear_cost_.GetCoefficients();
        }

        /**
         * Returns the gradient of the cost evaluated at x
         * @param x the input
         * @return grad f(x) = Ax + q
         */
        vectorx_t Gradient(const vectorx_t& x) const {
            return A_ * x + GetLinCoefficients();
        }

        /**
         * Returns the Hessian of the cost function evaluated at x
         * @param x the input
         * @return A
         */
        matrixx_t Hessian(const vectorx_t& x) const {
            return A_;
        }

        /**
         * The Hessian of a quadratic cost is always A
         * @return A
         */
        matrixx_t Hessian() const {
            return A_;
        }

    private:
        matrixx_t A_; // the coefficients of the linear cost
        LinearCost<scalar_t> linear_cost_ = LinearCost(vectorx_t(vectorx_t::Zero(0)), std::string(""));
    };
} // namespace torc::cost


#endif //TORC_QUADRATIC_COST_H
