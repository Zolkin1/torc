//
// Created by zolkin on 6/4/24.
//

#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"

#include "single_rigid_body.h"

#include <utility>

namespace torc::models {
    SingleRigidBody::SingleRigidBody(const std::string& name, const std::filesystem::path& urdf)
            : PinocchioModel(name, urdf) {
        system_type_ = HybridSystemNoImpulse;

        // Make the SRB model
        ref_config_ = pinocchio::neutral(pin_model_);    // Create neutral ref configuration

        MakeSingleRigidBody(ref_config_);
    }

    SingleRigidBody::SingleRigidBody(const std::string& name, const std::filesystem::path& urdf,
                                     const vectorx_t& ref_config)
            : PinocchioModel(name, urdf), ref_config_(ref_config) {
        system_type_ = HybridSystemNoImpulse;

        MakeSingleRigidBody(ref_config);

    }

    void SingleRigidBody::MakeSingleRigidBody(const vectorx_t& ref_config, bool reassign_full_model) {
        // Move the full pinocchio model
        if (reassign_full_model) {
            full_pin_model_ = pin_model_;
            full_pin_data_ = std::move(pin_data_);
        }

        // Lock every joint but the free-flyer
        std::vector<long unsigned int> joints_to_lock;
        int idx = 0;
        for (const auto& it : full_pin_model_.names) {
            if (it != "root_joint") {
                joints_to_lock.push_back(idx);
            }
            idx++;
        }

        pinocchio::buildReducedModel(full_pin_model_, joints_to_lock,
                                     ref_config, pin_model_);

        pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);
    }

    void SingleRigidBody::SetRefConfig(const vectorx_t& ref_config) {
        ref_config_ = ref_config;
        MakeSingleRigidBody(ref_config_, false);
    }

    vectorx_t SingleRigidBody::GetRefConfig() const {
        return ref_config_;
    }

    RobotStateDerivative SingleRigidBody::GetDynamics(const torc::models::RobotState& state,
                                                      const torc::models::PinocchioModel::vectorx_t& input) const {
        assert(state.q.size() == SRB_CONFIG_DIM);
        assert(state.v.size() == SRB_VEL_DIM);

        const vectorx_t& tau = InputsToTau(input);

        pinocchio::aba(pin_model_, *pin_data_, state.q, state.v, tau);

        RobotStateDerivative xdot(state.v, pin_data_->ddq);

        return xdot;
    }

    void SingleRigidBody::DynamicsDerivative(const RobotState& state,
                                             const vectorx_t& input,
                                             matrixx_t& A,
                                             matrixx_t& B) const {
        assert(state.q.size() == SRB_CONFIG_DIM);
        assert(state.v.size() == SRB_VEL_DIM);
        assert(A.rows() == GetDerivativeDim());
        assert(A.cols() == GetDerivativeDim());
        assert(B.rows() == GetDerivativeDim());
//        assert(B.cols() == act_mat_.cols());

        const vectorx_t& tau = InputsToTau(input);

        pinocchio::computeABADerivatives(pin_model_, *pin_data_, state.q, state.v, tau);

        A << matrixx_t::Zero(pin_model_.nv, pin_model_.nv), matrixx_t::Identity(pin_model_.nv, pin_model_.nv),
                pin_data_->ddq_dq, pin_data_->ddq_dv;

        // Make into a full matrix
        pin_data_->Minv.triangularView<Eigen::StrictlyLower>() =
                pin_data_->Minv.transpose().triangularView<Eigen::StrictlyLower>();

        B << matrixx_t::Zero(pin_model_.nv, input.size()), pin_data_->Minv * ActuationMapDerivative(input);
    }

    vectorx_t SingleRigidBody::InputsToTau(const vectorx_t& input) const {
        assert(input.size() % 6 == 0);
        Eigen::Vector<double, 6> tau = Eigen::Vector<double, 6>::Zero();

        const long pos_start = input.size()/2;
        const long num_pairs = input.size()/6;
        for (int i = 0; i < num_pairs; i++) {
            tau.segment<3>(0) += input.segment<3>(3*i);
            tau.segment<3>(3) += input.segment<3>(3*i).cross(input.segment<3>(pos_start + (3*i)));
        }

        return tau;
    }

    matrixx_t SingleRigidBody::ActuationMapDerivative(const vectorx_t& input,
                                                      bool force_and_pos) const {
        assert(input.size() % 6 == 0);
        const long pos_start = input.size()/2;
        const long num_pairs = input.size()/6;

        long input_vars = input.size();
        if (!force_and_pos) {
            input_vars = input.size()/2;
        }

        matrixx_t B(GetStateDim(), input_vars);

        if (!force_and_pos) {
            // Linear forces
            B.block(0, 0, 3, input_vars) =
                    matrixx_t::Ones(3, pos_start);

            // Torques
            // TODO
        } else {
            // Linear forces
            B.block(0, 0, 3, input_vars) << matrixx_t::Ones(3, pos_start),
                    matrixx_t::Zero(3, input_vars - pos_start);

            // Torques
            // TODO
        }
        return B;
    }

} // torc::models