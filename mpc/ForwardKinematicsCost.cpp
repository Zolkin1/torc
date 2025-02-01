//
// Created by zolkin on 1/31/25.
//

#include "pinocchio_interface.h"

#include "ForwardKinematicsCost.h"

namespace torc::mpc {
    ForwardKinematicsCost::ForwardKinematicsCost(int first_node, int last_node, const std::string &name,
        const vectorx_t &weights, const std::filesystem::path &deriv_lib_path, bool compile_derivs,
        const models::FullOrderRigidBody &model, const std::vector<std::string>& frames)
            : Cost(first_node, last_node, name, weights), model_(model), nq_(model.GetConfigDim()),
            nv_(model.GetVelDim()) {

        for (const auto& frame : frames) {
            cost_functions_.emplace(frame, std::make_unique<ad::CppADInterface>(
                std::bind(&ForwardKinematicsCost::CostFunction, this, frame, std::placeholders::_1,
                    std::placeholders::_2, std::placeholders::_3),
                name_ + "_" + frame + "_forward_kinematics_cost",
                deriv_lib_path,
                torc::ad::DerivativeOrder::FirstOrder, model.GetVelDim(), model.GetConfigDim() + 3 + weights.size(),
                compile_derivs
                ));
        }
    }

    std::pair<matrixx_t, vectorx_t> ForwardKinematicsCost::GetQuadraticApprox(const vectorx_t &x_lin, const vectorx_t &p,
        const std::string& frame) {
        if (x_lin.size() != nq_ || p.size() != 3) {
            throw std::runtime_error("[Cost Function] configuration approx reference or target has the wrong size!");
        }

        vectorx_t ad_p(cost_functions_[frame]->GetParameterSize());
        ad_p << x_lin, p, weights_;

        matrixx_t hessian_term;
        vectorx_t linear_term;

        matrixx_t jac;
        cost_functions_[frame]->GetGaussNewton(vectorx_t::Zero(nv_), ad_p, jac, hessian_term);
        hessian_term = 2*hessian_term;

        vectorx_t y;
        cost_functions_[frame]->GetFunctionValue(vectorx_t::Zero(nv_), ad_p, y);
        linear_term = 2*jac.transpose()*y;

        return {hessian_term, linear_term};
    }

    double ForwardKinematicsCost::GetCost(const std::string& frame, const vectorx_t &x, const vectorx_t &dx, const vectorx_t &p) {
        vectorx_t x_ad(cost_functions_[frame]->GetDomainSize());
        x_ad << dx;

        vectorx_t y;

        vectorx_t p_ad(cost_functions_[frame]->GetParameterSize());
        p_ad << x, p, weights_;

        cost_functions_[frame]->GetFunctionValue(x_ad, p_ad, y);
        return y.squaredNorm();
    }


    void ForwardKinematicsCost::CostFunction(const std::string& frame, const torc::ad::ad_vector_t &dq,
        const torc::ad::ad_vector_t &q_xyzdes_weight, torc::ad::ad_vector_t &frame_error) {

        const torc::ad::ad_vector_t& q = q_xyzdes_weight.head(nq_);
        const Eigen::Vector3<torc::ad::adcg_t>& des_pos = q_xyzdes_weight.segment<3>(nq_);

        // Get the current configuration
        const torc::ad::ad_vector_t q_curr = torc::models::ConvertdqToq(dq, q);

        // Get the frame location
        pinocchio::forwardKinematics(model_.GetADPinModel(), *model_.GetADPinData(), q_curr);
        const int frame_idx = model_.GetFrameIdx(frame);
        pinocchio::updateFramePlacement(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx);
        ad::ad_vector_t frame_pos = model_.GetADPinData()->oMf.at(frame_idx).translation();

        frame_error = frame_pos - des_pos;

        // Multiply by the weights
        for (int i = 0; i < 3; i++) {
            frame_error(i) = frame_error(i) * q_xyzdes_weight(nq_ + 3 + i);
        }
    }

}