#ifndef TORC_ANALYTIC_COST_H
#define TORC_ANALYTIC_COST_H

#include "linear_fn.h"

namespace torc::fn {
    /**
     * Class implementation of an analytic fn, where the differentials are provided by the user.
     * @tparam scalar_t the type of scalar used for the fn
     */
    template <class scalar_t>
    class AnalyticalFn: public BaseFn<scalar_t> {
        using matrixx_t = Eigen::MatrixX<scalar_t>;
        using vectorx_t = Eigen::VectorX<scalar_t>;

    public:
        /**
         * Constructor for the AnalyticalFn class
         * @param fn the fn function
         * @param grad the gradient of the function
         * @param hess the hessian of the function
         * @param dim the dimension of the function
         */
        AnalyticalFn(const std::function<scalar_t(vectorx_t)>& fn,
                     const std::function<vectorx_t(vectorx_t)>& grad,
                     const std::function<matrixx_t(vectorx_t)>& hess,
                     const size_t& dim,
                     const std::string& identifier="AnalyticCostInstance") {
            this->cost_ = fn;
            this->grad_ = grad;
            this->hess_ = hess;
            this->dim_ = dim;
            this->identifier_ = identifier;
        }


        /**
         * Evaluates the function at a point
         * @param x the input
         * @return f(x)
         */
        scalar_t Evaluate(const vectorx_t& x) const {
            return cost_(x);
        }


        /**
         * Evaluates the gradient of the function at a point
         * @param x the input
         * @return grad f(x)
         */
        vectorx_t Gradient(const vectorx_t& x) const {
            return grad_(x);
        }


        /**
         * Evaluates the Hessian of the function at a point
         * @param x the input
         * @return H_f(x)
         */
        matrixx_t Hessian(const vectorx_t& x) const {
            return hess_(x);
        }


    private:
        std::function<scalar_t(vectorx_t)> cost_;   // the original function
        std::function<vectorx_t(vectorx_t)> grad_;  // the gradient of the function
        std::function<matrixx_t(vectorx_t)> hess_;  // the hessian of the function
    };
}
#endif //TORC_ANALYTIC_COST_H
