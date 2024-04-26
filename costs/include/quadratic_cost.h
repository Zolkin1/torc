#ifndef TORC_QUADRATIC_COST_H
#define TORC_QUADRATIC_COST_H

#include <string>
#include "base_cost.h"
#include <iostream>

namespace torc {
    /**
     * Class implementation of a linear cost function, f(x) = x^T A x.
     * @tparam scalar_t the type of scalar used for the cost
     */
    template <class scalar_t>
    class QuadraticCost: public BaseCost<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrix_t = Eigen::MatrixX<scalar_t>;

    public:
        /**
         * Constructor for the quadratic cost class
         * @param coefficients the A in x^T A x.
         * @param identifier the name of the cost
         */
        QuadraticCost(const matrix_t& coefficients, const std::string& identifier) {
            if (coefficients.isUpperTriangular()) {
                this->A_ = coefficients.template selfadjointView<Eigen::Upper>();
            } else {
                if ((coefficients.transpose() - coefficients).squaredNorm() == 0) {
                    throw std::runtime_error("Quadratic cost: matrix must be either symmetric or upper triangular.");
                }
                this->A_ = coefficients;
            }
            this->identifier_ = identifier;
            this->domain_dim_ = static_cast<int>(coefficients.cols());
        }

        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return x^T A x
         */
        scalar_t Evaluate(const vectorx_t& x) const {
            return x.dot(A_ * x);
        }

        /**
         * Returns the A of the cost in upper triangular form.
         * @return the A_
         */
        matrix_t GetCoefficients() const {
            return A_;
        }

        /**
         * Returns the gradient of the cost evaluated at x
         * @param x the input
         * @return grad f(x) = (A + A^T) x
         */
        vectorx_t Gradient(const vectorx_t& x) const {
            return (A_ + A_.transpose()) * x;
        }

        /**
         * Returns the Hessian of the cost function evaluated at x
         * @param x the input
         * @return (A + A^T)
         */
        matrix_t Hessian(const vectorx_t& x) const {
            return A_ + A_.transpose();
        }

        /**
         * The Hessian of a quadratic cost is always (A + A^T)
         * @return (A + A^T)
         */
        matrix_t Hessian() const {
            return A_ + A_.transpose();
        }

    private:
        matrix_t A_; // the coefficients of the linear cost
    };
} // namespace torc

#endif //TORC_QUADRATIC_COST_H
