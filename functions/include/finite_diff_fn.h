#ifndef TORC_FINITE_DIFF_COST_H
#define TORC_FINITE_DIFF_COST_H

#include "explicit_fn.h"

namespace torc::fn {
    /**
     * Class implementation of a fn function whose differentials are evaluated using the finite difference method.
     * @tparam scalar_t the type of scalar used for the fn
     */
    template <class scalar_t>
    class FiniteDiffFn: public ExplicitFn<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;

    public:
        /**
         * Constructor for the Finite Difference Cost class
         * @param fn the fn function
         * @param dim the dimensions of the input of the fn function
         * @param grad_step the step used for the gradient, defaults to sqrt(ulp)
         * @param hess_step the step used for the hessian, defaults to sqrt(ulp_grad) = sqrt(sqrt(ulp))
         * @param identifier a string identifier of the fn function
         */
        FiniteDiffFn(const std::function<scalar_t(vectorx_t)>& fn,
                     const size_t& dim,
                     const scalar_t& grad_step=sqrt(std::numeric_limits<scalar_t>::epsilon()),
                     const scalar_t& hess_step=sqrt(sqrt(std::numeric_limits<scalar_t>::epsilon())),
                     const std::string& identifier="FiniteDifferenceCostInstance") {
            this->fn_ = fn;
            this->dim_ = dim;
//            this->grad_step_ = grad_step;
//            this->hess_step_ = hess_step;
            this->identifier_ = identifier;

            this->func_ = [fn] (const vectorx_t& x) { return fn(x); };

            std::function<vectorx_t(vectorx_t)> grad= [grad_step, dim, fn] (const vectorx_t& x) {
                matrixx_t perturbation = matrixx_t::Identity(dim, dim) * grad_step;
                vectorx_t grad(dim);
                for (unsigned i=0; i<dim; i++) {
                    const scalar_t pos_diff = fn(x + perturbation.col(i));
                    const scalar_t neg_diff = fn(x - perturbation.col(i));
                    grad(i) = (pos_diff - neg_diff) / (2 * grad_step);
                }
                return grad;
            };
            this->grad_ = grad;

            this->hess_ = [hess_step, dim, grad] (const vectorx_t& x) {
                matrixx_t perturbation = matrixx_t::Identity(dim, dim) * hess_step;

                matrixx_t hess(dim, dim);
                for (unsigned i=0; i<dim; i++) {
                    const vectorx_t pos_diff = grad(x + perturbation.col(i));
                    const vectorx_t neg_diff = grad(x - perturbation.col(i));
                    hess.col(i) = (pos_diff - neg_diff) / (2 * hess_step);
                }
                return hess;
            };
        }


//        /**
//         * Evaluates the fn function at a given point
//         * @param x the input to the function
//         * @return f(x)
//         */
//        scalar_t Evaluate(const vectorx_t& x) const {
//            return fn_(x);
//        }
//
//
//        /**
//         * Returns the gradient of the fn evaluated at x using the central finite difference method
//         * @param x the input
//         * @return grad f(x)
//         */
//        vectorx_t Gradient(const vectorx_t& x) const {
//            const size_t dim = this->dim_;
//            matrixx_t perturbation = matrixx_t::Identity(dim, dim) * this->grad_step_;
//
//            vectorx_t grad(dim);
//            for (unsigned i=0; i<dim; i++) {
//                const scalar_t pos_diff = this->fn_(x + perturbation.col(i));
//                const scalar_t neg_diff = this->fn_(x - perturbation.col(i));
//                grad(i) = (pos_diff - neg_diff) / (2 * this->grad_step_);
//            }
//            return grad;
//        }
//
//
//        /**
//         * Calculates the Hessian of the fn function evaluated at x using the central finite difference method
//         * @param x the input
//         * @return H_f(x)
//         */
//        matrixx_t Hessian(const vectorx_t& x) const {
//            const size_t dim = this->dim_;
//            matrixx_t perturbation = matrixx_t::Identity(dim, dim) * this->hess_step_;
//
//            matrixx_t hess(dim, dim);
//            for (unsigned i=0; i<dim; i++) {
//                const vectorx_t pos_diff = this->Gradient(x + perturbation.col(i));
//                const vectorx_t neg_diff = this->Gradient(x - perturbation.col(i));
//                hess.col(i) = (pos_diff - neg_diff) / (2 * this->hess_step_);
//            }
//            return hess;
//        }

//    private:
//        std::function<scalar_t(vectorx_t)> fn_;
//        scalar_t grad_step_;
//        scalar_t hess_step_;
    };
}

#endif //TORC_FINITE_DIFF_COST_H
