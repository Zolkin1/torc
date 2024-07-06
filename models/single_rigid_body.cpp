//
// Created by zolkin on 6/4/24.
//

#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"

#include "single_rigid_body.h"

#include <utility>

namespace torc::models {
    SingleRigidBody::SingleRigidBody(const std::string& name, const std::filesystem::path& urdf, int max_contacts)
            : PinocchioModel(name, urdf), max_contacts_(max_contacts) {
        system_type_ = HybridSystemNoImpulse;
        n_input_ = max_contacts_*6;

        // Make the SRB model
        ref_config_ = pinocchio::neutral(pin_model_);    // Create neutral ref configuration

        MakeSingleRigidBody(ref_config_);
    }

    SingleRigidBody::SingleRigidBody(const std::string& name, const std::filesystem::path& urdf,
                                     const vectorx_t& ref_config, int max_contacts)
            : PinocchioModel(name, urdf), ref_config_(ref_config), max_contacts_(max_contacts) {
        system_type_ = HybridSystemNoImpulse;
        n_input_ = max_contacts_*6;

        MakeSingleRigidBody(ref_config);
    }

    void SingleRigidBody::MakeSingleRigidBody(const vectorx_t& ref_config, bool reassign_full_model) {
        // Move the full pinocchio model
        if (reassign_full_model) {
            full_pin_model_ = pinocchio::Model(pin_model_);
            full_pin_data_ = std::move(pin_data_);
        }

        // Lock every joint but the free-flyer
        std::vector<long unsigned int> joints_to_lock;
        int idx = 0;
        for (const auto& it : full_pin_model_.names) {
            if (it != "root_joint" && it != "universe") {
                joints_to_lock.push_back(idx);
            }
            idx++;
        }

//        pin_model_ = pinocchio::Model();

        pin_model_ = pinocchio::buildReducedModel(full_pin_model_, joints_to_lock,
                                     ref_config);

        pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);
    }

    void SingleRigidBody::SetRefConfig(const vectorx_t& ref_config) {
        ref_config_ = ref_config;
        MakeSingleRigidBody(ref_config_, false);
    }

    vectorx_t SingleRigidBody::GetRefConfig() const {
        return ref_config_;
    }

    vectorx_t SingleRigidBody::GetDynamics(const vectorx_t& state,
                                           const vectorx_t& input) {
        vectorx_t q, v;
        ParseState(state, q, v);
        assert(q.size() == SRB_CONFIG_DIM);
        assert(v.size() == SRB_VEL_DIM);

        const vectorx_t& tau = InputsToTau(input);

        pinocchio::aba(pin_model_, *pin_data_, q, v, tau);
        vectorx_t xdot (v.size() + pin_data_->ddq.size());
        // RobotStateDerivative xdot(state.v, pin_data_->ddq);

        return xdot;
    }

    void SingleRigidBody::DynamicsDerivative(const vectorx_t &state,
                                             const vectorx_t &input,
                                             matrixx_t &A, matrixx_t &B) {
        vectorx_t q, v;
        ParseState(state, q, v);
        assert(A.rows() == GetDerivativeDim());
        assert(A.cols() == GetDerivativeDim());
        assert(B.rows() == GetDerivativeDim());
        assert(B.cols() == GetNumInputs());

        const vectorx_t &tau = InputsToTau(input);

        pinocchio::computeABADerivatives(pin_model_, *pin_data_, q, v, tau);

        A << matrixx_t::Zero(pin_model_.nv, pin_model_.nv),
            matrixx_t::Identity(pin_model_.nv, pin_model_.nv), pin_data_->ddq_dq,
            pin_data_->ddq_dv;

        // Make into a full matrix
        pin_data_->Minv.triangularView<Eigen::StrictlyLower>() =
            pin_data_->Minv.transpose().triangularView<Eigen::StrictlyLower>();

        matrixx_t act_map_deriv = ActuationMapDerivative(input);

        B << matrixx_t::Zero(pin_model_.nv, input.size()),
            pin_data_->Minv * act_map_deriv;
    }


    void SingleRigidBody::ParseState(const vectorx_t &state, vectorx_t &q,
                                     vectorx_t &v) {
        q = state.topRows(SRB_CONFIG_DIM);
        v = state.bottomRows(SRB_VEL_DIM);
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

        matrixx_t B(pin_model_.nv, input_vars);
        B.setZero();

        // Linear forces
        B.block(0, 0, 3, pos_start) =
                matrixx_t::Ones(3, pos_start);

        if (!force_and_pos) {
            // Torques
            // TODO
        } else {
            // Torques
            // TODO
        }

        return B;
    }

} // torc::models