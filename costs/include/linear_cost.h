#ifndef TORC_LINEAR_COST_H
#define TORC_LINEAR_COST_H

#include "base_cost.h"


namespace torc::cost {
    /**
     * Class implementation of a linear cost function, f(x) = q^T x
     * @tparam scalar_t the type of scalar used for the cost
     */
    template<class scalar_t>
    class LinearCost : public BaseCost<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;

    public:
        /**
         * Overloaded constructor for the LinearCost class.
         * @param dim input dimension, defaults coefficients to 0
         * @param identifier string identifier
         */
        explicit LinearCost(const int& dim, const std::string &identifier="Linear cost") {
            q_ = vectorx_t::Zero(dim);
            this->identifier_ = identifier;
            this->dim_ = dim;
        }

        /**
         * Overloaded constructor for the LinearCost class.
         * @param coefficients the linear coefficients
         * @param identifier string identifier
         */
        explicit LinearCost(const vectorx_t &coefficients, const std::string &identifier="Linear cost") {
            q_ = coefficients;
            this->identifier_ = identifier;
            this->dim_ = coefficients.size();
        }

        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return q^T x
         */
        scalar_t Evaluate(const vectorx_t &x) const {
            return q_.dot(x);
        }

        /**
         * Returns the q_ of the cost
         * @return the q_ q
         */
        vectorx_t GetCoefficients() const {
            return q_;
        }

        /**
         * Returns the gradient of the cost evaluated at x
         * @param x the input
         * @return grad f(x) = q
         */
        vectorx_t Gradient(const vectorx_t &x) const {
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
            return matrixx_t::Zero(this->dim_, this->dim_);
        }

        /**
         * The Hessian of a linear function is zero everywhere
         * @return a square zero matrix of dimension dim(x)
         */
        matrixx_t Hessian() const {
            return matrixx_t::Zero(this->dim_, this->dim_);
        }

    private:
        vectorx_t q_; // the coefficients of the linear cost
    };
} // namespace torc::cost


#endif //TORC_LINEAR_COST_H
