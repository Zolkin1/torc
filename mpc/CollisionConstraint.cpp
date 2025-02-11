//
// Created by zolkin on 1/27/25.
//

#include "CollisionConstraint.h"

#include "pinocchio_interface.h"

namespace torc::mpc {
    CollisionConstraint::CollisionConstraint(int first_node, int last_node, const std::string &name,
        const models::FullOrderRigidBody &model, const std::filesystem::path &deriv_lib_path, bool compile_derivs,
        const std::vector<CollisionData> &collision_data)
            : Constraint(first_node, last_node, name),  collision_data_(collision_data), model_(model) {

        config_dim_ = model.GetConfigDim();

        for (int i = 0; i < collision_data.size(); i++) {
            collision_function_.emplace_back(std::make_unique<ad::CppADInterface>(
                   std::bind(&CollisionConstraint::CollisionFunction, this,
                       collision_data_[i].frame1, collision_data_[i].frame2, std::placeholders::_1,
                       std::placeholders::_2, std::placeholders::_3),
                  name_ + "_" + collision_data_[i].frame1 + "_" + collision_data_[i].frame2 + "_collision_constraint",
                  deriv_lib_path,
                  ad::DerivativeOrder::FirstOrder, model_.GetVelDim(),  model_.GetConfigDim() + 2,
                  compile_derivs));
        }
    }

    std::pair<matrixx_t, vectorx_t> CollisionConstraint::GetLinearization(const vectorx_t &q_lin,
        int collision_idx) {
        vectorx_t x_zero = vectorx_t::Zero(collision_function_[collision_idx]->GetDomainSize());
        vectorx_t p(collision_function_[collision_idx]->GetParameterSize());
        p << q_lin, collision_data_[collision_idx].r1, collision_data_[collision_idx].r2;

        matrixx_t jac;
        collision_function_[collision_idx]->GetJacobian(x_zero, p, jac);

        vectorx_t y;
        collision_function_[collision_idx]->GetFunctionValue(x_zero, p, y);

        return {jac, y};
    }

    int CollisionConstraint::GetNumConstraints() const {
        return collision_function_.size()*collision_function_[0]->GetRangeSize();
    }

    int CollisionConstraint::GetNumCollisions() const {
        return collision_data_.size();
    }


    void CollisionConstraint::CollisionFunction(const std::string& frame1, const std::string& frame2,
        const ad::ad_vector_t& dqk, const ad::ad_vector_t& qk_r1_r2, ad::ad_vector_t& violation) {

        const ad::ad_vector_t& qk = qk_r1_r2.head(config_dim_);
        const ad::ad_vector_t q = models::ConvertdqToq(dqk, qk);
        const ad::adcg_t& r1 = qk_r1_r2(config_dim_);
        const ad::adcg_t& r2 = qk_r1_r2(config_dim_ + 1);

        pinocchio::forwardKinematics(model_.GetADPinModel(), *model_.GetADPinData(), q);
        const long frame_idx1 = model_.GetFrameIdx(frame1);
        pinocchio::updateFramePlacement(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx1);

        const long frame_idx2 = model_.GetFrameIdx(frame2);
        pinocchio::updateFramePlacement(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx2);

        ad::ad_vector_t frame_pos1 = model_.GetADPinData()->oMf.at(frame_idx1).translation();
        ad::ad_vector_t frame_pos2 = model_.GetADPinData()->oMf.at(frame_idx2).translation();

        violation.resize(1);
        violation(0) = (frame_pos1 - frame_pos2).norm() - (r1 + r2);
    }

    vectorx_t CollisionConstraint::GetViolation(const vectorx_t &q, const vectorx_t &dq, int collision_idx) {
        vectorx_t x(collision_function_[collision_idx]->GetDomainSize());
        x << dq;

        vectorx_t p(collision_function_[collision_idx]->GetParameterSize());
        p << q, collision_data_[collision_idx].r1, collision_data_[collision_idx].r2;

        vectorx_t violation;
        collision_function_[collision_idx]->GetFunctionValue(x, p, violation);

        return violation;
    }


}
