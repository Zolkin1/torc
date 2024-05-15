#ifndef TORC_BASE_COST_H
#define TORC_BASE_COST_H

#include <cstdint>
#include <string>
#include <eigen3/Eigen/Dense>


namespace torc::fn {

    /**
     * Abstract class representing a fn function to be optimized.
     * @tparam scalar_t the type of scalar used for the fn
     */
    template<class scalar_t>
    class BaseFn {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;

    public:
        /**
         * Evaluates the fn function at a given point
         * @param x the input to the function
         * @return the output of the function
         */
        virtual scalar_t Evaluate(const vectorx_t &x) const = 0;

        /**
         * Evaluates the gradient of the function at a given point.
         * @param x the input to the gradient of the function
         * @return the gradient of the function
         */
        virtual vectorx_t Gradient(const vectorx_t &x) const = 0;

        /**
         * Evaluates the Hessian of the function at a given point.
         * @param x the input to the Hessian of the function
         * @return the Hessian of the function
         */
        virtual matrixx_t Hessian(const vectorx_t &x) const = 0;

        bool IsValidIdentifier(const std::string &str) {
            if (!isalpha(str[0]) && str[0] != '_') {
                return false;
            }
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
        std::string identifier_ = "BaseCostInstance";   // the name assigned to this function
        size_t dim_ = 0;                                // the function domain's dimension
    };
}


#endif //TORC_BASE_COST_H
