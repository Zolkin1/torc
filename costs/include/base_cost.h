#ifndef TORC_BASE_COST_H
#define TORC_BASE_COST_H

#include <cstdint>
#include <string>
#include <eigen3/Eigen/Dense>

namespace torc {
    /**
     * Abstract class representing a cost function to be optimized.
     * @tparam scalar_t the type of scalar used for the cost
     */
    template <class scalar_t>
    class BaseCost {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;
    public:
        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return the output of the function
         */
        virtual scalar_t Evaluate(const vectorx_t& x) const = 0;

        /**
         * Evaluates the gradient of the function at a given point.
         * @param x the input to the gradient of the function
         * @return the gradient of the function
         */
        virtual vectorx_t Gradient(const vectorx_t& x) const = 0;

        /**
         * Evaluates the Hessian of the function at a given point.
         * @param x the input to the Hessian of the function
         * @return the Hessian of the function
         */
        virtual matrixx_t Hessian(const vectorx_t& x) const = 0;

        /**
         * Returns the identifier_ of the function
         * @return the function's name
         */
        std::string GetIdentifier() const { return identifier_; }

        /**
         * Returns the domain's dimension of the function
         * @return the domain's dimension
         */
        size_t GetDomainDim() const { return domain_dim_; }

    protected:
        std::string identifier_; // the (not necessarily unique) name assigned to this function
        size_t domain_dim_{};      // the function domain's dimension
    };
} // namespace torc

#endif //TORC_BASE_COST_H
