//
// Created by zolkin on 1/18/25.
//

#include "hpipm_mpc.h"

namespace torc::mpc {
    HpipmMpc::HpipmMpc(MpcSettings settings)
        : settings_(std::move(settings)){
        qp.resize(settings_.nodes + 1);
    }

    void HpipmMpc::SetDynamicsConstraints(std::vector<DynamicsConstraint> constraints) {
        dynamics_constraints_ = std::move(constraints);
        if (dynamics_constraints_.size() !=2) {
            throw std::runtime_error("For now we only accept exactly 2 dynamics constraints!");
        }
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
            if (node >= friction_cone_->GetFirstNode() && node < friction_cone_->GetLastNode()) {

            }

            // Swing height
            if (node >= swing_height_->GetFirstNode() && node < swing_height_->GetLastNode()) {

            }

            // Holonomic
            if (node >= holonomic_->GetFirstNode() && node < holonomic_->GetLastNode()) {

            }

            // Collision
            if (node >= collision_->GetFirstNode() && node < collision_->GetLastNode()) {

            }

            // Polytope
            if (node >= polytope_->GetFirstNode() && node < polytope_->GetLastNode()) {

            }
        }
    }


}