//
// Created by gavin on 4/26/2024.
//

#ifndef TORC_FINITE_DIFF_COST_H
#define TORC_FINITE_DIFF_COST_H

#include "base_cost.h"

namespace torc {
    template <class scalar_t>
    class FiniteDiffCost: public BaseCost<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;
    public:
        FiniteDiffCost(const std::function<scalar_t(vectorx_t)>& cost_fn,
                       const std::string& identifier="Finite difference cost.") {
            this->identifier_ = identifier;
            this->cost_fn_ = cost_fn;
        }
        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return q^T x
         */
        scalar_t Evaluate(const vectorx_t& x) const {
            return cost_fn_(x);
        }

        /**
         * Returns the gradient of the cost evaluated at x
         * @param x the input
         * @return grad f(x) = q
         */
        vectorx_t Gradient(const vectorx_t& x) const {
            return q_;
        }

        /**
         * The gradient of a linear function is constant everywhere, so we don't need an input
         * @return grad f(x) = q for all x
         */
        vectorx_t Gradient() const {
            return q_;
        }

        /**
         * The Hessian of a linear function is zero everywhere
         * @param x the input
         * @return a square zero matrix of dimension dim(x)
         */
        matrixx_t Hessian(const vectorx_t& x) const {
            return matrixx_t::Zero(this->domain_dim_, this->domain_dim_);
        }
    private:
        std::function<scalar_t(vectorx_t)> cost_fn_;
    };
}

#endif //TORC_FINITE_DIFF_COST_H
