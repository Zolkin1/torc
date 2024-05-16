#ifndef TORC_QUADRATIC_FN_H
#define TORC_QUADRATIC_FN_H

#include "explicit_fn.h"

namespace torc::fn {
    /**
     * Class implementation of a quadratic fn function, f(x) = (1/2) x^T A x + q^T x, where A is a symmetric matrix.
     * @tparam scalar_t the type of scalar used for the fn
     */
    template <class scalar_t>
    class QuadraticFn: public ExplicitFn<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;

    public:
        /**
         * Overloaded constructor for the QuadraticFn class.
         * @param quad_coefficients matrix coefficients (A) for f(x) = (1/2) x^T A x + q^T x, must be symmetric
         * @param lin_coefficients linear coefficients (q) for f(x)
         * @param name string name
         */
        explicit QuadraticFn(const matrixx_t& quad_coefficients,
                             const vectorx_t& lin_coefficients,
                             const std::string& name="QuadraticFnInstance") {
            if ((quad_coefficients.transpose() - quad_coefficients).squaredNorm() != 0) {
                throw std::runtime_error("Matrix must be symmetric.");
            }
            this->SetName(name);
            this->A_ = quad_coefficients;
            this->q_ = lin_coefficients;

            this->dim_ = lin_coefficients.size();
            this->func_ = [this](const vectorx_t& x) { return x.dot(this->A_ * x) * 0.5 + (this->q_).dot(x); };
            this->grad_ = [this](const vectorx_t& x) { return this->A_ * x + this->q_; };
            this->hess_ = [this](const vectorx_t& x) { return this->A_; };
        }


        /**
         * Overloaded constructor for the QuadraticFn class. The linear component defaults to 0.
         * @param coefficients matrix coefficients for f(x) = (1/2) x^T A x, must be symmetric
         * @param name string name
         */
        explicit QuadraticFn(const matrixx_t& coefficients,
                             const std::string& name="QuadraticFnInstance")
                   : QuadraticFn(coefficients,
                                 vectorx_t::Zero(coefficients.cols()),
                                 name) {}


        /**
         * Overloaded constructor for the QuadraticFn class.
         * @tparam dim input dimension
         * @param coefficients an upper triangular view (A) of the coefficients. The full matrix is constructed by
         *                     A^T + A, while the diagonal remains unchanged
         * @param lin_coefficients the linear coefficients, defaults to 0
         * @param name string name
         */
        template <int dim>
        explicit QuadraticFn(const Eigen::TriangularView<Eigen::Matrix<scalar_t, dim, dim>, Eigen::Upper>& coefficients,
                             const vectorx_t& lin_coefficients=vectorx_t::Zero(dim),
                             const std::string& name="QuadraticFnInstance")
                : QuadraticFn(matrixx_t(matrixx_t(coefficients).template selfadjointView<Eigen::Upper>()),
                              lin_coefficients,
                              name) {}


        /**
         * Get the full coefficient matrix of the function.
         * @return the A in f(x) = (1/2) x^T A x + q^T x
         */
        matrixx_t GetQuadCoefficients() const { return this->A_; }


        /**
         * Get the linear coefficients of the function
         * @return the q in (1/2) x^T A x + q^T x
         */
        vectorx_t GetLinCoefficients() const { return this->q_; }

    private:
        matrixx_t A_; // the coefficients of the quadratic function
        vectorx_t q_; // coefficients of the linear function
    };
} // namespace torc::fn

#endif //TORC_QUADRATIC_FN_H
