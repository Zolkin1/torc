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
        FiniteDiffCost(const std::function<scalar_t(vectorx_t)>& cost_fn,
                       const size_t& dim,
                       const std::string& identifier="Finite difference cost.") {
            this->identifier_ = identifier;
            this->cost_fn_ = cost_fn;
            this->domain_dim_ = dim;
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
            const double STEP = 3e-8;   // approximately sqrt(ulp)
            matrixx_t perturbation = Eigen::MatrixXd::Identity(dim, dim) * STEP;

            vectorx_t grad(dim);
            for (unsigned i=0; i<dim; i++) {
                scalar_t pos_diff = this->cost_fn_(x + perturbation.col(i));
                scalar_t neg_diff = this->cost_fn_(x - perturbation.col(i));
                grad(i) = (pos_diff - neg_diff) / (2 * STEP);
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
            const double STEP = 1e-3;   // approximately sqrt(step for grad)
            matrixx_t perturbation = Eigen::MatrixXd::Identity(dim, dim) * STEP;

            matrixx_t hess(dim, dim);
            for (unsigned i=0; i<dim; i++) {
                vectorx_t pos_diff = this->Gradient(x + perturbation.col(i));
                vectorx_t neg_diff = this->Gradient(x - perturbation.col(i));
                hess.col(i) = (pos_diff - neg_diff) / (2 * STEP);
            }
            return hess;
        }
    private:
        std::function<scalar_t(vectorx_t)> cost_fn_;
    };
}

#endif //TORC_FINITE_DIFF_COST_H
