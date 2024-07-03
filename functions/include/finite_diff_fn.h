#ifndef TORC_FINITE_DIFF_FN_H
#define TORC_FINITE_DIFF_FN_H

#include "explicit_fn.h"

namespace torc::fn {
    /**
     * @brief Class implementation of a function whose differentials are evaluated using the finite difference method.
     * @tparam scalar_t the type of scalar used for the fn
     */
    template <class scalar_t>
    class FiniteDiffFn: public ExplicitFn<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;

    public:
        /**
         * @brief Constructor for the Finite Difference class
         * @param func the function
         * @param dim the dimensions of the input of the func function
         * @param grad_step the step used for the gradient, defaults to sqrt(ulp)
         * @param hess_step the step used for the hessian, defaults to sqrt(ulp_grad) = sqrt(sqrt(ulp))
         * @param identifier a string identifier of the func function
         */
        FiniteDiffFn(const std::function<scalar_t(vectorx_t)>& func,
                     const size_t& dim,
                     const scalar_t& grad_step=sqrt(std::numeric_limits<scalar_t>::epsilon()),
                     const scalar_t& hess_step=sqrt(sqrt(std::numeric_limits<scalar_t>::epsilon())),
                     const std::string& identifier="FiniteDifferenceFnInstance") {
            this->dim_ = dim;
            this->SetName(identifier);

            this->func_ = func;

            this->grad_ = [this, grad_step] (const vectorx_t& x) {
                const size_t dim = this->dim_;
                matrixx_t perturbation = matrixx_t::Identity(dim, dim) * grad_step;
                vectorx_t grad(dim);
                for (unsigned i=0; i<dim; i++) {
                    const scalar_t pos_diff = this->func_(x + perturbation.col(i));
                    const scalar_t neg_diff = this->func_(x - perturbation.col(i));
                    grad(i) = (pos_diff - neg_diff) / (2 * grad_step);
                }
                return grad;
            };

            this->hess_ = [this, hess_step] (const vectorx_t& x) {
                const size_t dim = this->dim_;
                matrixx_t perturbation = matrixx_t::Identity(dim, dim) * hess_step;

                matrixx_t hess(dim, dim);
                for (unsigned i=0; i<dim; i++) {
                    const vectorx_t pos_diff = this->grad_(x + perturbation.col(i));
                    const vectorx_t neg_diff = this->grad_(x - perturbation.col(i));
                    hess.col(i) = (pos_diff - neg_diff) / (2 * hess_step);
                }
                return hess;
            };
        }
    };
}

#endif //TORC_FINITE_DIFF_FN_H
