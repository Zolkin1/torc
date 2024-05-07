//
// Created by gavin on 4/26/2024.
//

#ifndef TORC_FINITE_DIFF_COST_H
#define TORC_FINITE_DIFF_COST_H

#include "base_cost.h"
#include <iostream>

namespace torc::cost {
    /**
     * Class implementation of a cost function whose differentials are evaluated using the finite difference method.
     * @tparam scalar_t the type of scalar used for the cost
     */
    template <class scalar_t>
    class FiniteDiffCost: public BaseCost<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;
    public:
        /**
         * Constructor for the Finite Difference Cost class
         * @param fn the cost function
         * @param dim the dimensions of the input of the cost function
         * @param grad_step the step used for the gradient
         * @param hess_step the step used for the hessian
         * @param identifier a string identifier of the cost function
         */
        FiniteDiffCost(const std::function<scalar_t(vectorx_t)>& fn,
                       const size_t& dim,
                       const scalar_t& grad_step=3e-8,   // approximately sqrt(ulp)
                       const scalar_t& hess_step=1e-3,   // approximately sqrt(grad_step)
                       const std::string& identifier="Finite_Difference_Cost_Instance") {
            this->fn_ = fn;
            this->dim_ = dim;
            this->grad_step_ = grad_step;
            this->hess_step_ = hess_step;
            this->identifier_ = identifier;
        }

        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return f(x)
         */
        scalar_t Evaluate(const vectorx_t& x) const {
            return fn_(x);
        }

        /**
         * Returns the gradient of the cost evaluated at x using the central finite difference method
         * @param x the input
         * @return grad f(x)
         */
        vectorx_t Gradient(const vectorx_t& x) const {
            const size_t dim = this->dim_;
            matrixx_t perturbation = matrixx_t::Identity(dim, dim) * this->grad_step_;

            vectorx_t grad(dim);
            for (unsigned i=0; i<dim; i++) {
                const scalar_t pos_diff = this->fn_(x + perturbation.col(i));
                const scalar_t neg_diff = this->fn_(x - perturbation.col(i));
                grad(i) = (pos_diff - neg_diff) / (2 * this->grad_step_);
            }
            return grad;
        }

        /**
         * Calculates the Hessian of the cost function evaluated at x using the central finite difference method
         * @param x the input
         * @return H_f(x)
         */
        matrixx_t Hessian(const vectorx_t& x) const {
            const size_t dim = this->dim_;
            matrixx_t perturbation = matrixx_t::Identity(dim, dim) * this->hess_step_;

            matrixx_t hess(dim, dim);
            for (unsigned i=0; i<dim; i++) {
                const vectorx_t pos_diff = this->Gradient(x + perturbation.col(i));
                const vectorx_t neg_diff = this->Gradient(x - perturbation.col(i));
                hess.col(i) = (pos_diff - neg_diff) / (2 * this->hess_step_);
            }
            return hess;
        }

    private:
        std::function<scalar_t(vectorx_t)> fn_;
        scalar_t grad_step_;
        scalar_t hess_step_;
    };
}

#endif //TORC_FINITE_DIFF_COST_H
