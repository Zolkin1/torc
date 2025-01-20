//
// Created by zolkin on 1/18/25.
//

#include "hpipm_mpc.h"

#include <torc_timer.h>

// TODO: Can I have changing sizes every solve without causing slow downs? Then I may be able to remove some constraints (swing related)
namespace torc::mpc {
    HpipmMpc::HpipmMpc(MpcSettings settings, const models::FullOrderRigidBody& model)
        : settings_(std::move(settings)), model_(model) {
        qp.resize(settings_.nodes + 1); // Need the extra node for the model boundary
        solution_.resize(settings_.nodes + 1);

        nq_ = model_.GetConfigDim();
        nv_ = model_.GetVelDim();
        ntau_ = model_.GetVelDim() - FLOATING_VEL;
        nforces_ = settings_.num_contact_locations * CONTACT_3DOF;

        first_constraint_gen_ = true;

        qp_settings = settings_.qp_settings;

        // TODO: Adjust size to NOT account for the boundary (I think)
        traj_.UpdateSizes(nq_, nv_, ntau_, settings_.contact_frames, settings.nodes);
        in_contact_.resize(settings_.nodes);
        swing_traj_.resize(settings_.nodes);
        for (int i = 0; i < settings_.nodes; i++) {
            in_contact_[i] = 1;
            swing_traj_[i] = 0;
            vectorx_t q = model_.GetNeutralConfig();
            q(2) = 0.8;
            traj_.SetConfiguration(i, q);
        }

        traj_.SetDtVector(settings_.dt);

        // TODO: Get target values from user
        v_target_ = vectorx_t::Zero(nv_);
        tau_target_ = vectorx_t::Zero(ntau_);
        for (int i = 0; i < settings_.num_contact_locations; i++) {
            force_target_.emplace_back((vector3_t() << 0, 0, model_.GetMass()*9.81/2.0).finished());   // TODO: make a better force target
        }
        q_target_ = traj_.GetConfiguration(0);
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

    void HpipmMpc::SetVelTrackingCost(LinearLsCost cost) {
        vel_tracking_ = std::make_unique<LinearLsCost>(std::move(cost));
    }

    void HpipmMpc::SetTauTrackingCost(LinearLsCost cost) {
        tau_tracking_ = std::make_unique<LinearLsCost>(std::move(cost));
    }

    void HpipmMpc::SetForceTrackingCost(LinearLsCost cost) {
        force_tracking_ = std::make_unique<LinearLsCost>(std::move(cost));
    }

    void HpipmMpc::SetConfigTrackingCost(ConfigTrackingCost cost) {
        config_tracking_ = std::make_unique<ConfigTrackingCost>(std::move(cost));
    }

    void HpipmMpc::UpdateSetttings(MpcSettings settings) {
        settings_ = std::move(settings);
    }

    // TODO: Make sure there are no input constraint in the last node
    void HpipmMpc::CreateConstraints() {
        // Maybe the first time I should set all the sizes for hpipm
        if (first_constraint_gen_) {
            if (!config_box_ || !vel_box_ || !tau_box_ || !friction_cone_ || !swing_constraint_ || !holonomic_) {
                throw std::runtime_error("[HpipmMpc] a required constraint was not added!");
            }
            SetSizes();
            first_constraint_gen_ = false;
        }

        int traj_idx = 0;
        for (int node = 0; node < settings_.nodes + 1; node++) {
            std::cerr << "node: " << node << std::endl;

            vectorx_t force(CONTACT_3DOF*settings_.num_contact_locations);
            int force_idx = 0;
            for (const auto& frame : settings_.contact_frames) {
                force.segment<CONTACT_3DOF>(force_idx) = traj_.GetForce(traj_idx, frame);
                force_idx += CONTACT_3DOF;
            }

            int box_x_idx = 0;
            int box_u_idx = 0;
            int ineq_row_idx = 0;

            // Dynamics Constraints
            if (node < settings_.nodes) {
                // TODO: verify all of these mats
                if (dynamics_constraints_[0].IsInNodeRange(node)) {
                    std::cerr << "Adding FO dynamics..." << std::endl;
                    std::tie(qp[node].A, qp[node].B) = dynamics_constraints_[0].GetLinDynamics(
                        traj_.GetConfiguration(traj_idx), traj_.GetConfiguration(traj_idx + 1), traj_.GetVelocity(traj_idx),
                        traj_.GetVelocity(traj_idx + 1), traj_.GetTau(traj_idx), force, settings_.dt[traj_idx]);

                } else if (node == boundary_node_) {
                    std::cerr << "Adding boundary dynamics..." << std::endl;
                    std::tie(qp[node].A, qp[node].B) = dynamics_constraints_[0].GetBoundaryDynamics();
                } else {
                    std::cerr << "Adding ROM dynamics..." << std::endl;
                    // TODO: Error here
                    std::tie(qp[node].A, qp[node].B) = dynamics_constraints_[1].GetLinDynamics(
                                        traj_.GetConfiguration(traj_idx), traj_.GetConfiguration(traj_idx + 1), traj_.GetVelocity(traj_idx),
                                        traj_.GetVelocity(traj_idx + 1), traj_.GetTau(traj_idx), force, settings_.dt[node]);

                    // TODO: Check these
                    // std::cerr << "A:\n" << qp[node].A << std::endl;
                    // std::cerr << "B:\n" << qp[node].B << std::endl;
                }

                qp[node].b.setZero();
            }

            // Config box constraints
            if ((node >= config_box_->GetFirstNode() && node < config_box_->GetLastNode() + 1) && node != boundary_node_) {
                // std::cerr << "Adding config box..." << std::endl;
                // Set box indexes
                const auto& idxs = config_box_->GetIdxs();

                for (int i = 0; i < idxs.size(); i++) {
                    qp[node].idxbx[i] = idxs[i];
                }
                box_x_idx += idxs.size();

                // Bounds
                qp[node].lbx.head(idxs.size()) = config_box_->GetLowerBound(traj_.GetConfiguration(traj_idx));
                qp[node].ubx.head(idxs.size()) = config_box_->GetUpperBound(traj_.GetConfiguration(traj_idx));
            }

            // Vel box constraints
            if ((node >= vel_box_->GetFirstNode() && node < vel_box_->GetLastNode() + 1) && node != boundary_node_) {
                // std::cerr << "Adding vel box..." << std::endl;
                // Full order model this is in the state
                if (dynamics_constraints_[0].IsInNodeRange(node)) {
                    // Set box indexes
                    const auto& idxs = vel_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbx[i + box_x_idx] = idxs[i];
                    }

                    // Bounds
                    qp[node].lbx.tail(idxs.size()) = vel_box_->GetLowerBound(traj_.GetVelocity(traj_idx));
                    qp[node].ubx.tail(idxs.size()) = vel_box_->GetUpperBound(traj_.GetVelocity(traj_idx));
                } else {    // Reduced order model this is in the input
                    // Set box indexes
                    const auto& idxs = vel_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbu[i] = idxs[i];
                    }

                    box_u_idx += idxs.size();

                    // Bounds
                    qp[node].lbu.head(idxs.size()) = vel_box_->GetLowerBound(traj_.GetVelocity(traj_idx));
                    qp[node].ubu.head(idxs.size()) = vel_box_->GetUpperBound(traj_.GetVelocity(traj_idx));
                }
            }

            // Torque box constraints
            if ((node >= tau_box_->GetFirstNode() && node < tau_box_->GetLastNode() + 1) && node != boundary_node_) {
                // std::cerr << "Adding tau box..." << std::endl;
                if (dynamics_constraints_[0].IsInNodeRange(node)) {
                    // Set box indexes
                    const auto& idxs = tau_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbu[i] = idxs[i];
                    }

                    // Bounds
                    qp[node].lbu.head(idxs.size()) = tau_box_->GetLowerBound(traj_.GetTau(traj_idx));
                    qp[node].ubu.head(idxs.size()) = tau_box_->GetUpperBound(traj_.GetTau(traj_idx));
                }
            }

            // Friction cone constraints
            if ((node >= friction_cone_->GetFirstNode() && node < friction_cone_->GetLastNode() + 1) && node != boundary_node_) {
                // std::cerr << "Adding friction cone..." << std::endl;
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
                // std::cerr << "Adding swing..." << std::endl;
                for (const auto& frame : settings_.contact_frames) {
                    const auto [c_block, y_segment] =
                        swing_constraint_->GetLinearization(traj_.GetConfiguration(traj_idx), swing_traj_[traj_idx], frame);
                    // y_segment.size should = 1
                    qp[node].C.block(ineq_row_idx, 0, y_segment.size(), nq_) = in_contact_[traj_idx]*c_block;
                    qp[node].lg(ineq_row_idx) = -in_contact_[traj_idx]*y_segment(0);
                    qp[node].ug(ineq_row_idx) = -in_contact_[traj_idx]*y_segment(0);
                    ineq_row_idx += y_segment.size();
                }
            }

            // Holonomic
            if ((node >= holonomic_->GetFirstNode() && node < holonomic_->GetLastNode() + 1) && node != boundary_node_) {
                // std::cerr << "Adding holonomic..." << std::endl;
                for (const auto& frame : settings_.contact_frames) {
                    const auto [jac, y_segment] =
                        holonomic_->GetLinearization(traj_.GetConfiguration(traj_idx), traj_.GetVelocity(traj_idx), frame);
                    if (dynamics_constraints_[0].IsInNodeRange(traj_idx)) {
                        qp[node].C.middleRows(ineq_row_idx, y_segment.size()) = in_contact_[traj_idx]*jac;
                    } else {
                        qp[node].C.middleRows(ineq_row_idx, y_segment.size()) =
                            in_contact_[node]*jac.leftCols(nq_ + FLOATING_VEL);
                        qp[node].D.block(ineq_row_idx, 0, y_segment.size(), FLOATING_VEL) =
                            in_contact_[node]*jac.rightCols(FLOATING_VEL);
                    }
                    qp[node].lg.segment<2>(ineq_row_idx) = -in_contact_[traj_idx]*y_segment;
                    qp[node].ug.segment<2>(ineq_row_idx) = -in_contact_[traj_idx]*y_segment;

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

            if (node != boundary_node_) {
                traj_idx++;
            }
        }
    }

    // TODO: Make sure there are only state costs in the last node
    void HpipmMpc::CreateCost() {
        if (!vel_tracking_ || !tau_tracking_ || !force_tracking_) {
            throw std::runtime_error("[HpipmMpc] Required cost not set!");
        }

        int traj_idx = 0;
        for (int node = 0; node < settings_.nodes + 1; node++) {

            if (vel_tracking_->IsInNodeRange(node) && node != boundary_node_) {
                const auto [hess, lin]
                    = vel_tracking_->GetQuadraticApprox(traj_.GetVelocity(traj_idx), GetVelocityTarget(traj_idx));

                if (dynamics_constraints_[0].IsInNodeRange(node)) {
                    qp[node].Q.bottomRightCorner(nv_, nv_) = hess;
                    qp[node].q.tail(nv_) = lin;
                } else {
                    qp[node].Q.bottomRightCorner<FLOATING_VEL, FLOATING_VEL>() = hess.topRightCorner<FLOATING_VEL, FLOATING_VEL>();
                    qp[node].q.tail<FLOATING_VEL>() = lin.head<FLOATING_VEL>();
                    qp[node].R.topLeftCorner(ntau_, ntau_) = hess.bottomLeftCorner(ntau_, ntau_);
                    qp[node].r.head(ntau_) = lin.tail(ntau_);
                }
            }

            if (tau_tracking_->IsInNodeRange(node) && node != boundary_node_ && dynamics_constraints_[0].IsInNodeRange(node)) {
                const auto [hess, lin]
                    = tau_tracking_->GetQuadraticApprox(traj_.GetTau(traj_idx), GetTauTarget(traj_idx));

                qp[node].R.topLeftCorner(ntau_, ntau_) = hess;
                qp[node].r.head(ntau_) = lin;
            }

            if (force_tracking_->IsInNodeRange(node) && node != boundary_node_) {
                for (int i = 0; i < settings_.contact_frames.size(); i++) {
                    const auto [hess, lin]
                        = force_tracking_->GetQuadraticApprox(
                            traj_.GetForce(traj_idx, settings_.contact_frames[i]), GetForceTarget(traj_idx, i));
                }
            }

            if (config_tracking_->IsInNodeRange(node) && node != boundary_node_) {
                const auto [hess, lin] =
                    config_tracking_->GetQuadraticApprox(traj_.GetConfiguration(traj_idx), GetConfigTarget(traj_idx));
                qp[node].Q.topLeftCorner(nv_, nv_) = hess;
                qp[node].q.head(nv_) = lin;
            }

            if (node != boundary_node_) {
                traj_idx++;
            }
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
            qp[node].lbx = vectorx_t::Zero(nx_box);
            qp[node].ubx = vectorx_t::Zero(nx_box);

            qp[node].idxbu.resize(nu_box);
            qp[node].lbu = vectorx_t::Zero(nu_box);
            qp[node].ubu = vectorx_t::Zero(nu_box);

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


    void HpipmMpc::Compute(const vectorx_t &q0, const vectorx_t &v0, Trajectory& traj_out) {
        NanCheck(); // TODO: Can remove

        traj_.SetConfiguration(0, q0);
        traj_.SetVelocity(0, v0);

        torc::utils::TORCTimer timer;
        timer.Tic();
        const auto res = solver_->solve(vectorx_t::Zero(model_.GetConfigDim() + model_.GetVelDim()),
            qp, solution_); // TODO: Might need to remove this x0 input
        timer.Toc();

        const auto stats = solver_->getSolverStatistics();

        std::cout << "Res: " << res << std::endl;
        std::cout << stats << std::endl;
        std::cout << "solve time: " << timer.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl;

        ConvertQpSolToTraj();
        traj_out = traj_;
        traj_.ExportToCSV(std::filesystem::current_path() / "trajectory_output.csv");
        //TODO: Visualize trajectory!
    }

    void HpipmMpc::ConvertQpSolToTraj() {
        int traj_idx = 0;
        for (int node = 0; node < settings_.nodes + 1; node++) {

            std::cout << "node: " << node << std::endl;

            if (dynamics_constraints_[0].IsInNodeRange(node)) {
                // std::cout << "q before: " << traj_.GetConfiguration(node).transpose() << std::endl;
                traj_.SetConfiguration(traj_idx, models::ConvertdqToq<double>(solution_[node].x.head(nv_), traj_.GetConfiguration(node)));
                // std::cout << "dq: " << solution_[node].x.head(nv_).transpose() << std::endl;

                traj_.SetVelocity(traj_idx, traj_.GetVelocity(node) + solution_[node].x.tail(nv_));
                traj_.SetTau(traj_idx, traj_.GetTau(node) + solution_[node].u.head(ntau_));
            } else if (node != boundary_node_ && node < settings_.nodes) {
                traj_.SetConfiguration(traj_idx, models::ConvertdqToq<double>(solution_[node].x.head(nv_), traj_.GetConfiguration(node)));

                vectorx_t v(nv_);
                v << solution_[node].x.tail<FLOATING_VEL>(), solution_[node].u.head(ntau_);
                traj_.SetVelocity(traj_idx, traj_.GetVelocity(node) + v);
            }

            if (node != boundary_node_ && node < settings_.nodes) {
                for (int i = 0; i < settings_.contact_frames.size(); i++) {
                    traj_.SetForce(traj_idx, settings_.contact_frames[i],
                        traj_.GetForce(traj_idx, settings_.contact_frames[i]) + solution_[node].u.segment<CONTACT_3DOF>(ntau_ + i*3));
                }
            }

            if (node != boundary_node_) {
                traj_idx++;
            }
        }
    }

    // --------- Get Targets --------- //
    vectorx_t HpipmMpc::GetVelocityTarget(int node) const {
        return v_target_;
    }

    vectorx_t HpipmMpc::GetTauTarget(int node) const {
        return tau_target_;
    }

    vectorx_t HpipmMpc::GetForceTarget(int node, int force_idx) const {
        return force_target_[force_idx];
    }

    vectorx_t HpipmMpc::GetConfigTarget(int node) const {
        return q_target_;
    }

    // --------- Set Targets --------- //
    void HpipmMpc::SetVelTarget(const vectorx_t &v_target) {
        v_target_ = v_target;
    }

    void HpipmMpc::SetConfigTarget(const vectorx_t &q_target) {
        q_target_ = q_target;
    }



}

