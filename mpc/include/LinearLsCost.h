//
// Created by zolkin on 1/20/25.
//

#ifndef LINEARLS_H
#define LINEARLS_H

#include "Cost.h"
#include "cpp_ad_interface.h"

namespace torc::mpc {
    class LinearLsCost : public Cost {
    public:
        LinearLsCost(int first_node, int last_node, const std::string& name, const vectorx_t& weights,
            const std::filesystem::path& deriv_lib_path, bool compile_derivs);

        /**
         * @brief
         * @param x_lin point to take the approximation around
         * @param p parameters
         * @return
         */
        std::pair<matrixx_t, vectorx_t> GetQuadraticApprox(const vectorx_t& x_lin, const vectorx_t& p);

    protected:
        void CostFunction(const torc::ad::ad_vector_t& dx,
                            const torc::ad::ad_vector_t& xref_xtarget_weight,
                            torc::ad::ad_vector_t& x_diff) const;

        std::unique_ptr<ad::CppADInterface> cost_function_;
        int var_size_;
    private:
    };
}


#endif //LINEARLS_H
