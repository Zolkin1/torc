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
        for (int i = 0; i < settings_.nodes; i++) {
            // Dynamics
            if (i <= dynamics_constraints_[0].GetFirstNode() && i < dynamics_constraints_[0].GetLastNode()) {
                // std::tie(qp[i].A, qp[i].B) = dynamics_constraints_[0].GetLinDynamics();
            } else if (i <= dynamics_constraints_[1].GetFirstNode() && i < dynamics_constraints_[1].GetLastNode()) {
                // std::tie(qp[i].A, qp[i].B) = dynamics_constraints_[0].GetLinDynamics();
            } else {
                throw std::runtime_error("[HpipmMpc] dynamics constraint nodes are not consistent!");
            }
            qp[i].b.setZero();  // TODO: Do I need to set the size

            // State Constraints

            // Input Constraints
        }
    }


}