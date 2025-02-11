//
// Created by zolkin on 1/31/25.
//

#ifndef FORWARDKINEMATICSCOST_H
#define FORWARDKINEMATICSCOST_H

#include <filesystem>
#include "full_order_rigid_body.h"
#include "Cost.h"

namespace torc::mpc {
    class ForwardKinematicsCost : public Cost {
    public:
        ForwardKinematicsCost(int first_node, int last_node, const std::string& name, int var_size,
            const std::filesystem::path& deriv_lib_path, bool compile_derivs, const models::FullOrderRigidBody& model,
            const std::vector<std::string>& frames);

        std::pair<matrixx_t, vectorx_t> GetQuadraticApprox(const vectorx_t& x_lin, const vectorx_t& p, const vectorx_t& weight,
            const std::string& frame);

        double GetCost(const std::string& frame, const vectorx_t& x, const vectorx_t& dx, const vectorx_t& p,
            const vectorx_t& weight);

    protected:
        void CostFunction(const std::string& frame, const torc::ad::ad_vector_t &dq,
            const torc::ad::ad_vector_t &q_xyzdes_weight, torc::ad::ad_vector_t &frame_error);

        int nq_;
        int nv_;
        std::map<std::string, std::unique_ptr<ad::CppADInterface>> cost_functions_;
        models::FullOrderRigidBody model_;

    private:
    };
}


#endif //FORWARDKINEMATICSCOST_H
