//
// Created by zolkin on 1/18/25.
//

#include "hpipm_mpc.h"

// TODO: Can I have changing sizes every solve without causing slow downs? Then I may be able to remove some constraints (swing related)
namespace torc::mpc {
    HpipmMpc::HpipmMpc(MpcSettings settings, const models::FullOrderRigidBody& model)
        : settings_(std::move(settings)), model_(model) {
        qp.resize(settings_.nodes + 1); // Need the extra node for the model boundary
        solution.resize(settings_.nodes + 1);

        nq_ = model_.GetConfigDim();
        nv_ = model_.GetVelDim();
        ntau_ = model_.GetVelDim() - FLOATING_VEL;
        nforces_ = settings_.num_contact_locations * CONTACT_3DOF;

        first_solve_ = true;

        // TODO: Set solver settings
        // qp_settings.mode

        traj_.UpdateSizes(nq_, nv_, ntau_, settings_.contact_frames, settings.nodes + 1);
        in_contact_.resize(settings_.nodes + 1);
        swing_traj_.resize(settings_.nodes + 1);
        for (int i = 0; i < settings_.nodes + 1; i++) {
            in_contact_[i] = 1;
            swing_traj_[i] = 0;
            traj_.SetConfiguration(i, model_.GetNeutralConfig());
        }
    }

    void HpipmMpc::SetDynamicsConstraints(std::vector<DynamicsConstraint> constraints) {
        dynamics_constraints_ = std::move(constraints);
        if (dynamics_constraints_.size() !=2) {
            throw std::runtime_error("For now we only accept exactly 2 dynamics constraints!");
        }

        if (dynamics_constraints_[0].GetFirstNode() != 0) {
            throw std::runtime_error("First dynamics constraint must start at node = 0!");
        }
        if (dynamics_constraints_[0].GetLastNode() != dynamics_constraints_[1].GetFirstNode()) {
            throw std::runtime_error("First and last nodes for the dynamics constraints must match!");
        }
        if (dynamics_constraints_[1].GetLastNode() != settings_.nodes) {
            throw std::runtime_error("Last dynamics node must match the settings!");
        }

        boundary_node_ = dynamics_constraints_[0].GetLastNode();
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

    // TODO: Make sure there are no input constraint in the last node
    void HpipmMpc::CreateConstraints() {
        // Maybe the first time I should set all the sizes for hpipm
        if (first_solve_) {
            if (!config_box_ || !vel_box_ || !tau_box_ || !friction_cone_ || !swing_constraint_ || !holonomic_) {
                throw std::runtime_error("[HpipmMpc] a required constraint was not added!");
            }
            SetSizes();
            first_solve_ = false;
        }

        for (int node = 0; node < settings_.nodes + 1; node++) {
            std::cerr << "node: " << node << std::endl;

            vectorx_t force(CONTACT_3DOF*settings_.num_contact_locations);
            int force_idx = 0;
            for (const auto& frame : settings_.contact_frames) {
                force.segment<CONTACT_3DOF>(force_idx) = traj_.GetForce(node, frame);
                force_idx += CONTACT_3DOF;
            }

            int box_x_idx = 0;
            int box_u_idx = 0;
            int ineq_row_idx = 0;

            // Dynamics Constraints
            if (node < settings_.nodes) {
                // TODO: verify all of these mats
                if (dynamics_constraints_[0].IsInNodeRange(node)) {
                    std::tie(qp[node].A, qp[node].B) = dynamics_constraints_[0].GetLinDynamics(
                        traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1), traj_.GetVelocity(node),
                        traj_.GetVelocity(node + 1), traj_.GetTau(node), force, settings_.dt[node]);
                } else if (node == boundary_node_) {
                    std::tie(qp[node].A, qp[node].B) = dynamics_constraints_[0].GetBoundaryDynamics();
                } else {
                    // TODO: Error here
                    std::tie(qp[node].A, qp[node].B) = dynamics_constraints_[1].GetLinDynamics(
                                        traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1), traj_.GetVelocity(node),
                                        traj_.GetVelocity(node + 1), traj_.GetTau(node), force, settings_.dt[node]);
                    // TODO: Check these - check against finite differencing the actual Forward Dynamics
                    // std::cerr << "A:\n" << qp[node].A << std::endl;
                    // std::cerr << "B:\n" << qp[node].B << std::endl;
                }

                qp[node].b.setZero();
            }

            // Config box constraints
            if ((node >= config_box_->GetFirstNode() && node < config_box_->GetLastNode() + 1) && node != boundary_node_) {
                // Set box indexes
                const auto& idxs = config_box_->GetIdxs();

                for (int i = 0; i < idxs.size(); i++) {
                    qp[node].idxbx[i] = idxs[i];
                }
                box_x_idx += idxs.size();

                // Bounds
                qp[node].lbx.head(idxs.size()) = config_box_->GetLowerBound(traj_.GetConfiguration(node));
                qp[node].ubx.head(idxs.size()) = config_box_->GetUpperBound(traj_.GetConfiguration(node));
            }

            // Vel box constraints
            if ((node >= vel_box_->GetFirstNode() && node < vel_box_->GetLastNode() + 1) && node != boundary_node_) {
                // Full order model this is in the state
                if (dynamics_constraints_[0].IsInNodeRange(node)) {
                    // Set box indexes
                    const auto& idxs = vel_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbx[i + box_x_idx] = idxs[i];
                    }

                    // Bounds
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
                    qp[node].lbu.head(idxs.size()) = vel_box_->GetLowerBound(traj_.GetVelocity(node));
                    qp[node].ubu.head(idxs.size()) = vel_box_->GetUpperBound(traj_.GetVelocity(node));
                }
            }

            // Torque box constraints
            if ((node >= tau_box_->GetFirstNode() && node < tau_box_->GetLastNode() + 1) && node != boundary_node_) {
                if (dynamics_constraints_[0].IsInNodeRange(node)) {
                    // Set box indexes
                    const auto& idxs = tau_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbu[i] = idxs[i];
                    }

                    // Bounds
                    qp[node].lbu.head(idxs.size()) = tau_box_->GetLowerBound(traj_.GetTau(node));
                    qp[node].ubu.head(idxs.size()) = tau_box_->GetUpperBound(traj_.GetTau(node));
                }
            }

            // Friction cone constraints
            if ((node >= friction_cone_->GetFirstNode() && node < friction_cone_->GetLastNode() + 1) && node != boundary_node_) {
                int col = ntau_;
                for (int contact = 0; contact < settings_.num_contact_locations; contact++) {
                    const auto [d_block, lg_segment]
                        = friction_cone_->GetLinearization(force.segment<3>(3*contact));
                    qp[node].D.block(contact, col, 1, CONTACT_3DOF) = d_block;
                    qp[node].lg.segment<1>(contact) = lg_segment;
                    qp[node].ug_mask(contact) = 0;
                    col += CONTACT_3DOF;
                    ineq_row_idx++;
                }
            }

            // Swing height
            if ((node >= swing_constraint_->GetFirstNode() && node < swing_constraint_->GetLastNode() + 1) && node != boundary_node_) {
                for (const auto& frame : settings_.contact_frames) {
                    const auto [c_block, y_segment] =
                        swing_constraint_->GetLinearization(traj_.GetConfiguration(node), swing_traj_[node], frame);
                    // y_segment.size should = 1
                    qp[node].C.block(ineq_row_idx, 0, y_segment.size(), nq_) = in_contact_[node]*c_block;
                    qp[node].lg(ineq_row_idx) = -in_contact_[node]*y_segment(0);
                    qp[node].ug(ineq_row_idx) = -in_contact_[node]*y_segment(0);
                    ineq_row_idx += y_segment.size();
                }
            }

            // Holonomic
            if ((node >= holonomic_->GetFirstNode() && node < holonomic_->GetLastNode() + 1) && node != boundary_node_) {
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

            // // Collision
            // if (collision_->IsInNodeRange(node)) {
            //
            // }
            //
            // // Polytope
            // if (polytope_->IsInNodeRange(node)) {
            //
            // }
        }
    }

    // TODO: Make sure there are only state costs in the last node
    void HpipmMpc::CreateCost() {
        for (int node = 0; node < settings_.nodes + 1; node++) {
            // TODO: For now sub in something simple
            qp[node].R.setIdentity();
            qp[node].Q.setIdentity();
        }
    }


    void HpipmMpc::SetSizes() {
        // Resize all the QP mats and set them to zero.
        for (int node = 0; node < settings_.nodes + 1; node++) {
            int nx1, nx2;
            int nu = ntau_ + CONTACT_3DOF*settings_.num_contact_locations;
            int nx_box, nu_box;
            int n_other_constraints = 0;
            if (friction_cone_->IsInNodeRange(node)) {
                n_other_constraints += settings_.num_contact_locations*friction_cone_->GetNumConstraints();
            }
            if (swing_constraint_->IsInNodeRange(node)) {
                n_other_constraints += swing_constraint_->GetNumConstraints();
            }
            if (holonomic_->IsInNodeRange(node)) {
                n_other_constraints += holonomic_->GetNumConstraints();
            }
            // TODO: Add other constraints

            if (dynamics_constraints_[0].IsInNodeRange(node)) {
                nx1 = nv_ + nv_;
                nx2 = nx1;
                nx_box = nv_ + nv_ - (FLOATING_VEL + FLOATING_VEL);
                nu_box = ntau_;
            } else if (node == boundary_node_) {
                nx1 = nv_ + nv_;
                nx2 = nv_ + FLOATING_VEL;
                nx_box = 0;
                nu_box = 0;
            } else {
                nx1 = nv_ + FLOATING_VEL;
                nx2 = nx1;
                nx_box = nq_;
                nu_box = ntau_;
            }

            std::cout << "node: " << node << ", nx1: " << nx1 << ", nx2: " << nx2 << ", nu: " << nu << std::endl;

            if (node < settings_.nodes) {
                // Dynamics
                qp[node].A = matrixx_t::Zero(nx2, nx1);
                qp[node].B = matrixx_t::Zero(nx2, nu);
                qp[node].b = vectorx_t::Zero(nx1);
            }

            // Cost
            qp[node].Q = matrixx_t::Zero(nx1, nx1);
            qp[node].R = matrixx_t::Zero(nu, nu);
            qp[node].S = matrixx_t::Zero(nu, nx1);
            qp[node].q = vectorx_t::Zero(nx1);
            qp[node].r = vectorx_t::Zero(nu);

            // Box Constraints
            qp[node].idxbx.resize(nx_box);
            qp[node].lbx.resize(nx_box);
            qp[node].ubx.resize(nx_box);

            qp[node].idxbu.resize(nu_box);
            qp[node].lbu.resize(nu_box);
            qp[node].ubu.resize(nu_box);

            // Other Constraints
            qp[node].C = matrixx_t::Zero(n_other_constraints, nx1);
            qp[node].D = matrixx_t::Zero(n_other_constraints, nu);
            qp[node].lg = vectorx_t::Zero(n_other_constraints);
            qp[node].ug = vectorx_t::Zero(n_other_constraints);
            qp[node].ug_mask = vectorx_t::Ones(n_other_constraints);   // TODO: Determine if 1 is active constraint or not
        }

        solver_ = std::make_unique<hpipm::OcpQpIpmSolver>(qp, qp_settings);
    }

    void HpipmMpc::NanCheck() {
        for (int node = 0; node < settings_.nodes + 1; node++) {
            if (node < settings_.nodes) {
                if (qp[node].A.array().isNaN().any()) {
                    throw std::runtime_error("NaN in A matrix in node: " + std::to_string(node));
                }

                if (qp[node].B.array().isNaN().any()) {
                    throw std::runtime_error("NaN in B matrix in node: " + std::to_string(node));
                }

                if (qp[node].b.array().isNaN().any()) {
                    throw std::runtime_error("NaN in b matrix in node: " + std::to_string(node));
                }
            }

            if (qp[node].Q.array().isNaN().any()) {
                throw std::runtime_error("NaN in Q matrix in node: " + std::to_string(node));
            }

            if (qp[node].R.array().isNaN().any()) {
                throw std::runtime_error("NaN in R matrix in node: " + std::to_string(node));
            }

            if (qp[node].S.array().isNaN().any()) {
                throw std::runtime_error("NaN in S matrix in node: " + std::to_string(node));
            }

            if (qp[node].q.array().isNaN().any()) {
                throw std::runtime_error("NaN in q matrix in node: " + std::to_string(node));
            }

            if (qp[node].r.array().isNaN().any()) {
                throw std::runtime_error("NaN in r matrix in node: " + std::to_string(node));
            }

            if (qp[node].lbx.array().isNaN().any()) {
                throw std::runtime_error("NaN in lbx matrix in node: " + std::to_string(node));
            }

            if (qp[node].ubx.array().isNaN().any()) {
                throw std::runtime_error("NaN in ubx matrix in node: " + std::to_string(node));
            }

            if (qp[node].lbu.array().isNaN().any()) {
                throw std::runtime_error("NaN in lbu matrix in node: " + std::to_string(node));
            }

            if (qp[node].ubu.array().isNaN().any()) {
                throw std::runtime_error("NaN in ubu matrix in node: " + std::to_string(node));
            }
        }
    }


    void HpipmMpc::Compute(const vectorx_t &q0, const vectorx_t &v0) {
        NanCheck(); // TODO: Can remove

        const auto res = solver_->solve(vectorx_t::Zero(model_.GetConfigDim() + model_.GetVelDim()),
            qp, solution); // TODO: Might need to remove this x0 input

        std::cout << "Res: " << res << std::endl;

    }
}

