#ifndef TORC_QUADRATIC_COST_H
#define TORC_QUADRATIC_COST_H

#include "base_fn.h"
#include "linear_fn.h"

namespace torc::fn {
    /**
     * Class implementation of a quadratic fn function, f(x) = (1/2) x^T A x + q^T x, where A is a symmetric matrix.
     * @tparam scalar_t the type of scalar used for the fn
     */
    template <class scalar_t>
    class QuadraticFn: public BaseFn<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;

    public:
        /**
         * Overloaded constructor for the QuadraticFn class.
         * @param coefficients matrix coefficients (A) for f(x) = (1/2) x^T A x + q^T x, must be symmetric
         * @param lin_coefficients linear coefficients (q) for f(x)
         * @param identifier string identifier
         */
        explicit QuadraticFn(const matrixx_t& coefficients,
                             const vectorx_t& lin_coefficients,
                             const std::string& identifier="QuadraticCostInstance") {
            if ((coefficients.transpose() - coefficients).squaredNorm() != 0) {
                throw std::runtime_error("Matrix must be symmetric.");
            }
            this->A_ = coefficients;
            this->linear_cost_ = LinearFn<scalar_t>(lin_coefficients);
            this->identifier_ = identifier;
            this->dim_ = coefficients.cols();
        }


        /**
         * Overloaded constructor for the QuadraticFn class. The linear component defaults to 0.
         * @param coefficients matrix coefficients for f(x) = (1/2) x^T A x, must be symmetric
         * @param identifier string identifier
         */
        explicit QuadraticFn(const matrixx_t& coefficients,
                             const std::string& identifier="QuadraticCostInstance")
                   : QuadraticFn(coefficients,
                                 vectorx_t::Zero(coefficients.cols()),
                                 identifier) {}


        /**
         * Overloaded constructor for the QuadraticFn class.
         * @tparam dim input dimension
         * @param coefficients an upper triangular view (A) of the coefficients. The full matrix is constructed by
         *                     A^T + A, while the diagonal remains unchanged
         * @param lin_coefficients the linear coefficients, defaults to 0
         * @param identifier string identifier
         */
        template <int dim>
        explicit QuadraticFn(const Eigen::TriangularView<Eigen::Matrix<scalar_t, dim, dim>, Eigen::Upper>& coefficients,
                             const vectorx_t& lin_coefficients=vectorx_t::Zero(dim),
                             const std::string& identifier="QuadraticCostInstance")
                : QuadraticFn(matrixx_t(matrixx_t(coefficients).template selfadjointView<Eigen::Upper>()),
                              lin_coefficients,
                              identifier) {}


        /**
         * Evaluates the function at a given point
         * @param x the input to the function
         * @return (1/2) x^T A x + q^T x
         */
        scalar_t Evaluate(const vectorx_t& x) const {
            return x.dot(A_ * x) * 0.5 + linear_cost_.Evaluate(x);
        }

        /**
         * Get the full coefficient matrix of the function.
         * @return the A in f(x) = (1/2) x^T A x + q^T x
         */
        matrixx_t GetQuadCoefficients() const {
            return A_;
        }

        /**
         * Get the linear coefficients of the function
         * @return the q in (1/2) x^T A x + q^T x
         */
        vectorx_t GetLinCoefficients() const {
            return linear_cost_.GetCoefficients();
        }

        /**
         * Evaluates the gradient of the function evaluated at x
         * @param x the input
         * @return grad f(x) = Ax + q
         */
        vectorx_t Gradient(const vectorx_t& x) const {
            return A_ * x + GetLinCoefficients();
        }

        /**
         * Evaluates the Hessian of the function evaluated at x
         * @param x the input
         * @return A
         */
        matrixx_t Hessian(const vectorx_t& x) const {
            return A_;
        }

        /**
         * Evaluates the gradient of the function, which is always A
         * @return A
         */
        matrixx_t Hessian() const {
            return A_;
        }

    private:
        matrixx_t A_; // the coefficients of the quadratic fn
        LinearFn<scalar_t> linear_cost_ = LinearFn<scalar_t>(0);
    };
} // namespace torc::fn

#endif //TORC_QUADRATIC_COST_H
