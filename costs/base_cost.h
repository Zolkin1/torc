#ifndef TORC_BASE_COST_H
#define TORC_BASE_COST_H

#include <cstdint>
#include <string>
#include <eigen3/Eigen/Dense>

namespace torc {
    /**
     * Abstract class representing a cost function to be optimized.
     * @tparam dtype the type of scalar used for the cost
     */
    template <class dtype>
    class BaseCost {
    public:
        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return the output of the function
         */
        virtual dtype Evaluate(const Eigen::VectorX<dtype> &x) const = 0;
        /**
         * Evaluates the gradient of the function at a given point.
         * @param x the input to the gradient of the function
         * @return the gradient of the function
         */
        virtual Eigen::VectorX<dtype> Gradient(const Eigen::VectorX<dtype> &x) const = 0;
        /**
         * Evaluates the Hessian of the function at a given point.
         * @param x the input to the Hessian of the function
         * @return the Hessian of the function
         */
        virtual Eigen::MatrixX<dtype> Hessian(const Eigen::VectorX<dtype> &x) const = 0;
        /**
         * Returns the identifier of the function
         * @return the function's name
         */
        std::string GetIdentifier() const { return identifier; }
        /**
         * Returns the domain's dimension of the function
         * @return the domain's dimension
         */
        size_t GetDomainDim() const { return domain_dim; }
    protected:
        std::string identifier; // the (not necessarily unique) name assigned to this function
        size_t domain_dim;      // the function domain's dimension
    };
} // namespace torc

#endif //TORC_BASE_COST_H
