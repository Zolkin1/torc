#ifndef TORC_LINEAR_COST_H
#define TORC_LINEAR_COST_H

#include <string>
#include "base_cost.h"

namespace torc {
    /**
     * Class implementation of a linear cost function, f(x) = q^T x
     * @tparam dtype the type of scalar used for the cost
     */
    template <class dtype>
    class LinearCost: public BaseCost<dtype> {
      public:
        LinearCost(const Eigen::VectorX<dtype> &coefficients, const std::string &identifier) {
            q_ = coefficients;
            this->identifier_ = identifier;
            this->domain_dim_ = coefficients.size();
        }

        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return q^T x
         */
        dtype Evaluate(const Eigen::VectorX<dtype> &x) const {
            return q_.dot(x);
        }

        /**
         * Returns the q_ of the cost
         * @return the q_ q
         */
        Eigen::VectorX<dtype> GetCoefficients() const {
            return q_;
        }

        /**
         * Returns the gradient of the cost evaluated at x
         * @param x the input
         * @return grad f(x) = q
         */
        Eigen::VectorX<dtype> Gradient(const Eigen::VectorX<dtype> &x) const {
            return q_;
        }

        /**
         * The gradient of a linear function is constant everywhere, so we don't need an input
         * @return grad f(x) = q for all x
         */
        Eigen::VectorX<dtype> Gradient() const {
            return q_;
        }

        /**
         * The Hessian of a linear function is zero everywhere
         * @param x the input
         * @return a square zero matrix of dimension dim(x)
         */
        Eigen::MatrixX<dtype> Hessian(const Eigen::VectorX<dtype> &x) const {
            return Eigen::MatrixX<dtype>::Zero(this->domain_dim_, this->domain_dim_);
        }

      private:
        Eigen::VectorX<dtype> q_; // the q_ of the linear cost
    };
} // namespace torc

#endif //TORC_LINEAR_COST_H
