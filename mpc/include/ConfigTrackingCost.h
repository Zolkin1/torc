//
// Created by zolkin on 1/20/25.
//

#ifndef CONFIGTRACKINGCOST_H
#define CONFIGTRACKINGCOST_H
#include "NonlinearLsCost.h"
#include "full_order_rigid_body.h"


namespace torc::mpc {
    class ConfigTrackingCost : public Cost {
    public:
        ConfigTrackingCost(int start_node, int last_node, const std::string& name, int var_size,
            const std::filesystem::path& deriv_lib_path, bool compile_derivs, const models::FullOrderRigidBody& model);

        /**
         * @brief
         * @param x_lin point to take the approximation around
         * @param p parameters
         * @return
         */
        std::pair<matrixx_t, vectorx_t> GetQuadraticApprox(const vectorx_t& x_lin, const vectorx_t& p,
            const vectorx_t& weight);

        double GetCost(const vectorx_t& x, const vectorx_t& dx, const vectorx_t& p, const vectorx_t& weight) const;

    protected:
        void CostFunction(const torc::ad::ad_vector_t& dx, const torc::ad::ad_vector_t& xref_xtarget_weight,
            torc::ad::ad_vector_t& x_diff) const;

        std::unique_ptr<ad::CppADInterface> cost_function_;
        int var_size_;
        int nq_;
        int nv_;
    private:
    };
}


#endif //CONFIGTRACKINGCOST_H
