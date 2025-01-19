//
// Created by zolkin on 1/18/25.
//

#include "hpipm_mpc.h"

// TODO: Can I have changing sizes every solve without causing slow downs? Then I may be able to remove some constraints (swing related)
namespace torc::mpc {
    HpipmMpc::HpipmMpc(MpcSettings settings, const models::FullOrderRigidBody& model)
        : settings_(std::move(settings)), model_(model){
        qp.resize(settings_.nodes + 1);

        nq_ = model_.GetConfigDim();
        nv_ = model_.GetVelDim();
        ntau_ = model_.GetVelDim() - FLOATING_VEL;
        nforces_ = settings_.num_contact_locations * 3;

        SetSizes();
    }

    void HpipmMpc::SetDynamicsConstraints(std::vector<DynamicsConstraint> constraints) {
        dynamics_constraints_ = std::move(constraints);
        if (dynamics_constraints_.size() !=2) {
            throw std::runtime_error("For now we only accept exactly 2 dynamics constraints!");
        }
    }

    void HpipmMpc::SetConfigBox(const BoxConstraint& constraint) {
        config_box_ = std::make_unique<BoxConstraint>(constraint);
    }

    void HpipmMpc::SetVelBox(const BoxConstraint &constraints) {
        vel_box_ = std::make_unique<BoxConstraint>(constraints);
    }

    void HpipmMpc::SetTauBox(const BoxConstraint &constraints) {
        tau_box_ = std::make_unique<BoxConstraint>(constraints);
    }

    void HpipmMpc::SetFrictionCone(FrictionConeConstraint constraints) {
        friction_cone_ = std::make_unique<FrictionConeConstraint>(std::move(constraints));
    }

    void HpipmMpc::SetSwingConstraint(SwingConstraint constraints) {
        swing_constraint_ = std::make_unique<SwingConstraint>(std::move(constraints));
    }

    void HpipmMpc::SetHolonomicConstraint(HolonomicConstraint constraints) {
        holonomic_ = std::make_unique<HolonomicConstraint>(std::move(constraints));
    }

    void HpipmMpc::UpdateSetttings(MpcSettings settings) {
        settings_ = std::move(settings);
    }

    void HpipmMpc::CreateConstraints() {
        // Maybe the first time I should set all the sizes for hpipm

        for (int node = 0; node < settings_.nodes; node++) {
            vectorx_t force(3*settings_.num_contact_locations);
            int idx = 0;
            for (const auto& frame : settings_.contact_frames) {
                force.segment<3>(idx) = traj_.GetForce(node, frame);
                idx += 3;
            }

            int box_x_idx = 0;
            int box_u_idx = 0;
            int ineq_row_idx = 0;

            // Dynamics Constraints
            if (dynamics_constraints_[0].IsInNodeRange(node)) {
                std::tie(qp[node].A, qp[node].B) = dynamics_constraints_[0].GetLinDynamics(
                    traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1), traj_.GetVelocity(node),
                    traj_.GetVelocity(node + 1), traj_.GetTau(node), force, settings_.dt[node]);
            } else if (dynamics_constraints_[1].IsInNodeRange(node)) {
                std::tie(qp[node].A, qp[node].B) = dynamics_constraints_[0].GetLinDynamics(
                    traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1), traj_.GetVelocity(node),
                    traj_.GetVelocity(node + 1), traj_.GetTau(node), force, settings_.dt[node]);
            } else {
                throw std::runtime_error("[HpipmMpc] dynamics constraint nodes are not consistent!");
            }
            qp[node].b.setZero();  // TODO: Do I need to set the size?

            // Config box constraints
            if (node >= config_box_->GetFirstNode() && node < config_box_->GetLastNode()) {
                // Set box indexes
                const auto& idxs = config_box_->GetIdxs();
                for (int i = 0; i < idxs.size(); i++) {
                    qp[node].idxbx[i] = idxs[i];
                }
                box_x_idx += idxs.size();

                // Bounds
                // TODO: Requires the size to be set first!
                qp[node].lbx.head(idxs.size()) = config_box_->GetLowerBound(traj_.GetConfiguration(node));
                qp[node].ubx.head(idxs.size()) = config_box_->GetUpperBound(traj_.GetConfiguration(node));
            }

            // Vel box constraints
            if (vel_box_->IsInNodeRange(node)) {
                // Full order model this is in the state
                if (dynamics_constraints_[0].IsInNodeRange(node)) {
                    // Set box indexes
                    const auto& idxs = vel_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbx[i + box_x_idx] = idxs[i];
                    }

                    // Bounds
                    // TODO: Requires the size to be set first!
                    qp[node].lbx.tail(idxs.size()) = vel_box_->GetLowerBound(traj_.GetVelocity(node));
                    qp[node].ubx.tail(idxs.size()) = vel_box_->GetUpperBound(traj_.GetVelocity(node));
                } else {    // Reduced order model this is in the input
                    // Set box indexes
                    const auto& idxs = vel_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbu[i] = idxs[i];
                    }

                    box_u_idx += idxs.size();

                    // Bounds
                    // TODO: Requires the size to be set first!
                    qp[node].lbu.head(idxs.size()) = vel_box_->GetLowerBound(traj_.GetVelocity(node));
                    qp[node].ubu.head(idxs.size()) = vel_box_->GetUpperBound(traj_.GetVelocity(node));
                }
            }

            // Torque box constraints
            if (tau_box_->IsInNodeRange(node)) {
                if (dynamics_constraints_[0].IsInNodeRange(node)) {
                    // Set box indexes
                    const auto& idxs = tau_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbu[i + box_u_idx] = idxs[i];
                    }

                    // Bounds
                    // TODO: Requires the size to be set first!
                    qp[node].lbu.tail(idxs.size()) = tau_box_->GetLowerBound(traj_.GetTau(node));
                    qp[node].ubu.tail(idxs.size()) = tau_box_->GetUpperBound(traj_.GetTau(node));
                }
            }

            // Friction cone constraints
            if (friction_cone_->IsInNodeRange(node)) {
                int col = ntau_;
                for (int contact = 0; contact < settings_.num_contact_locations; contact++) {
                    const auto [d_block, lg_segment]
                        = friction_cone_->GetLinearization(force.segment<3>(3*contact));
                    qp[node].D.block(contact, col, 1, 3) = d_block;
                    qp[node].lg.segment<1>(contact) = lg_segment;
                    qp[node].ug_mask(contact) = 0;  // TODO: Verify this
                    col +=3;
                    ineq_row_idx++;
                }
            }

            // Swing height
            if (swing_constraint_->IsInNodeRange(node)) {
                for (const auto& frame : settings_.contact_frames) {
                    const auto [c_block, y_segment] =
                        swing_constraint_->GetLinearization(traj_.GetConfiguration(node), swing_traj_[node], frame);
                    // y_segment.size should = 3
                    qp[node].C.block(ineq_row_idx, 0, y_segment.size(), nq_) = in_contact_[node]*c_block;
                    qp[node].lg.segment<3>(ineq_row_idx) = -in_contact_[node]*y_segment;
                    qp[node].ug.segment<3>(ineq_row_idx) = -in_contact_[node]*y_segment;
                    ineq_row_idx += y_segment.size();
                }
            }

            // Holonomic
            if (holonomic_->IsInNodeRange(node)) {
                for (const auto& frame : settings_.contact_frames) {
                    const auto [jac, y_segment] =
                        holonomic_->GetLinearization(traj_.GetConfiguration(node), traj_.GetVelocity(node), frame);
                    if (dynamics_constraints_[0].IsInNodeRange(node)) {
                        qp[node].C.middleRows(ineq_row_idx, y_segment.size()) = in_contact_[node]*jac;
                    } else {
                        qp[node].C.middleRows(ineq_row_idx, y_segment.size()) =
                            in_contact_[node]*jac.leftCols(nq_ + FLOATING_VEL);
                        qp[node].D.block(ineq_row_idx, 0, y_segment.size(), FLOATING_VEL) =
                            in_contact_[node]*jac.rightCols(FLOATING_VEL);
                    }
                    qp[node].lg.segment<2>(ineq_row_idx) = -in_contact_[node]*y_segment;
                    qp[node].ug.segment<2>(ineq_row_idx) = -in_contact_[node]*y_segment;

                    ineq_row_idx += y_segment.size();
                }
            }

            // Collision
            if (collision_->IsInNodeRange(node)) {

            }

            // Polytope
            if (polytope_->IsInNodeRange(node)) {

            }
        }
    }

    void HpipmMpc::SetSizes() {
        // Resize all the QP mats and set them to zero.
        throw std::runtime_error("TODO: Impelement SetSizes!");
    }



}

