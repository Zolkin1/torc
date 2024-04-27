//
// Created by gavin on 4/26/2024.
//

#ifndef TORC_FINITE_DIFF_COST_H
#define TORC_FINITE_DIFF_COST_H

#include "base_cost.h"
#include <iostream>

namespace torc {
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
         *
         * @param cost_fn the cost function
         * @param dim   the dimensions of the input of the cost function
         * @param identifier a string identifier of the cost function
         */
        FiniteDiffCost(const std::function<scalar_t(vectorx_t)>& cost_fn,
                       const size_t& dim,
                       const scalar_t grad_step=3e-8,   // approximately sqrt(ulp)
                       const scalar_t hess_step=1e-3,   // approximately sqrt(grad_step)
                       const std::string& identifier="Finite difference cost.") {
            this->identifier_ = identifier;
            this->cost_fn_ = cost_fn;
            this->domain_dim_ = dim;
            this->grad_step_ = grad_step;
            this->hess_step_ = hess_step;
        }

        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return f(x)
         */
        scalar_t Evaluate(const vectorx_t& x) const {
            return cost_fn_(x);
        }

        /**
         * Returns the gradient of the cost evaluated at x using the central finite difference method
         * @param x the input
         * @return grad f(x)
         */
        vectorx_t Gradient(const vectorx_t& x) const {
            unsigned dim = this->domain_dim_;
            matrixx_t perturbation = Eigen::MatrixXd::Identity(dim, dim) * this->grad_step_;

            vectorx_t grad(dim);
            for (unsigned i=0; i<dim; i++) {
                scalar_t pos_diff = this->cost_fn_(x + perturbation.col(i));
                scalar_t neg_diff = this->cost_fn_(x - perturbation.col(i));
                grad(i) = (pos_diff - neg_diff) / (2 * this->grad_step_);
            }
            return grad;
        }

        /**
         * Calculates the Hessian of the cost function evaluated at x using the central finite difference method
         * @param x the input
         * @return Hf(x)
         */
        matrixx_t Hessian(const vectorx_t& x) const {
            unsigned dim = this->domain_dim_;
            matrixx_t perturbation = Eigen::MatrixXd::Identity(dim, dim) * this->hess_step_;

            matrixx_t hess(dim, dim);
            for (unsigned i=0; i<dim; i++) {
                vectorx_t pos_diff = this->Gradient(x + perturbation.col(i));
                vectorx_t neg_diff = this->Gradient(x - perturbation.col(i));
                hess.col(i) = (pos_diff - neg_diff) / (2 * this->hess_step_);
            }
            return hess;
        }
    private:
        std::function<scalar_t(vectorx_t)> cost_fn_;
        scalar_t grad_step_;
        scalar_t hess_step_;
    };
}

#endif //TORC_FINITE_DIFF_COST_H
