#ifndef TORC_EXPLICIT_DIFF_COST_H
#define TORC_EXPLICIT_DIFF_COST_H

#include "linear_cost.h"

namespace torc {
    template <class scalar_t>
    class ExplicitDifferentialCost: public BaseCost<scalar_t> {
        using matrixx_t = Eigen::MatrixX<scalar_t>;
        using vectorx_t = Eigen::MatrixX<scalar_t>;
    public:
        ExplicitDifferentialCost(std::function<scalar_t(vectorx_t)> cost,
                                 std::function<vectorx_t(vectorx_t)> grad,
                                 std::function<matrixx_t(vectorx_t)> hess) {

        }
    private:
        std::function<vectorx_t(vectorx_t)> grad;
        std::function<matrixx_t(vectorx_t)> hess;
    };
}
#endif //TORC_EXPLICIT_DIFF_COST_H
