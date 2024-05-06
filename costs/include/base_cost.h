#ifndef TORC_BASE_COST_H
#define TORC_BASE_COST_H

#include <cstdint>
#include <string>
#include <eigen3/Eigen/Dense>

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cppad/cg.hpp>
#include "base_cost.h"

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

        bool IsValidIdentifier(const std::string& str) {
            if (str[0] != '_' && !isalpha(str[0])) {
                return false;
            }
//            return false;
            return std::all_of(str.cbegin(), str.cend(), [](char c) {return (isalnum(c) || c=='_');});
        }

        /**
         * Returns the identifier_ of the function
         * @return the function's name
         */
        [[nodiscard]] std::string GetIdentifier() const { return identifier_; }

        /**
         * Returns the domain's dimension of the function
         * @return the domain's dimension
         */
        [[nodiscard]] size_t GetDomainDim() const { return dim_; }

    protected:
        std::string identifier_; // the (not necessarily unique) name assigned to this function
        size_t dim_;      // the function domain's dimension
    };
} // namespace torc

#endif //TORC_BASE_COST_H
