//
// Created by zolkin on 1/20/25.
//

#ifndef NONLINEARLSCOST_H
#define NONLINEARLSCOST_H

#include <filesystem>
#include "cpp_ad_interface.h"
#include "Cost.h"

namespace torc::mpc {
    // class NonlinearLsCost : public Cost {
    // public:
    //     NonlinearLsCost(int first_node, int last_node, const std::string& name, const vectorx_t& weights,
    //         const std::filesystem::path& deriv_lib_path, bool compile_derivs);
    //
    //     /**
    //      * @brief
    //      * @param x_lin point to take the approximation around
    //      * @param p parameters
    //      * @return
    //      */
    //     std::pair<matrixx_t, vectorx_t> GetQuadraticApprox(const vectorx_t& x_lin, const vectorx_t& p);
    //
    // protected:
    //     virtual void CostFunction(const torc::ad::ad_vector_t& dx,
    //         const torc::ad::ad_vector_t& xref_xtarget_weight,
    //         torc::ad::ad_vector_t& x_diff) const = 0;
    //
    //     ad::CppADInterface cost_function_;
    //     int var_size_;
    // private:
    // };
}


#endif //NONLINEARLSCOST_H
