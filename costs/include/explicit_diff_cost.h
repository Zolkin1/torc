#ifndef TORC_EXPLICIT_DIFF_COST_H
#define TORC_EXPLICIT_DIFF_COST_H

#include "linear_cost.h"

namespace torc {
    template <class scalar_t>
    class ExplicitDifferentialCost: public BaseCost<scalar_t> {
        using matrixx_t = Eigen::MatrixX<scalar_t>;
        using vectorx_t = Eigen::VectorX<scalar_t>;
    public:
        ExplicitDifferentialCost(const std::function<scalar_t(vectorx_t)>& cost,
                                 const std::function<vectorx_t(vectorx_t)>& grad,
                                 const std::function<matrixx_t(vectorx_t)>& hess,
                                 size_t dim) {
            this->cost_ = cost;
            this->grad_ = grad;
            this->hess_ = hess;
            this->domain_dim_ = dim;
        }

        scalar_t Evaluate(const vectorx_t& x) const {
            return cost_(x);
        }

        vectorx_t Gradient(const vectorx_t& x) const {
            return grad_(x);
        }

        matrixx_t Hessian(const vectorx_t& x) const {
            return hess_(x);
        }

    private:
        std::function<scalar_t(vectorx_t)> cost_;
        std::function<vectorx_t(vectorx_t)> grad_;
        std::function<matrixx_t(vectorx_t)> hess_;
    };
}
#endif //TORC_EXPLICIT_DIFF_COST_H
