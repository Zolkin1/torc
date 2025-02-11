//
// Created by zolkin on 1/18/25.
//

#include "hpipm_mpc.h"
#include <torc_timer.h>

// TODO: Can I have changing sizes every solve without causing slow downs? Then I may be able to remove some constraints (swing related)
namespace torc::mpc {
    HpipmMpc::HpipmMpc(MpcSettings settings, const models::FullOrderRigidBody& model)
        : settings_(std::move(settings)), model_(model),
        v_target_(model.GetVelDim(), settings.nodes),
        q_target_(model.GetConfigDim(), settings.nodes),
        tau_target_(model.GetVelDim() - FLOATING_VEL, settings.nodes),
        solve_counter_(0) {
        qp.resize(settings_.nodes); // Need the extra node for the model boundary
        solution_.resize(settings_.nodes);

        // TODO: Remove and read in from yaml
        settings_.swing_buffer = 0.004;

        nq_ = model_.GetConfigDim();
        nv_ = model_.GetVelDim();
        ntau_ = model_.GetVelDim() - FLOATING_VEL;
        nforces_ = settings_.num_contact_locations * CONTACT_3DOF;

        first_constraint_gen_ = true;

        qp_settings = settings_.qp_settings;

        traj_.UpdateSizes(nq_, nv_, ntau_, settings_.contact_frames, settings.nodes);
        int frame_idx = 0;
        for (const auto& frame : settings_.contact_frames) {
            in_contact_.insert({frame, {}});
            swing_traj_.insert({frame, {}});
            contact_info_.insert({frame, {}});
            end_effector_targets_.insert({frame, {}});
            for (int i = 0; i < settings_.nodes; i++) {
                in_contact_[frame].push_back(1);
                swing_traj_[frame].push_back(0);
                contact_info_[frame].push_back(ContactSchedule::GetDefaultContactInfo());
                end_effector_targets_[frame].emplace_back(settings_.hip_offsets[2*frame_idx], settings_.hip_offsets[2*frame_idx + 1], 0);
                // std::cout << "A:\n" << contact_info_[frame][i].A_ << std::endl;
            }
            frame_idx++;
        }

        for (int i = 0; i < settings_.nodes; i++) {
            vectorx_t q = model_.GetNeutralConfig();
            q(2) = 0.8;
            traj_.SetConfiguration(i, q);
            for (const auto& frame : settings_.contact_frames) {
                traj_.SetForce(i, frame, GetForceTarget(i, frame));
                traj_.SetInContact(i, frame, true);
            }
        }

        traj_.SetDtVector(settings_.dt);

        v_target_.SetAllData(vectorx_t::Zero(model_.GetVelDim()));
        tau_target_.SetAllData(vectorx_t::Zero(model_.GetVelDim() - FLOATING_VEL));
        q_target_.SetAllData(traj_.GetConfiguration(0));

        alpha_ = 1;

        for (int i = 0; i < settings_.cost_data.size(); i++) {
            if (settings_.cost_data[i].type == CostTypes::Configuration) {
                fo_config_idx_ = i;
            }
            if (settings_.cost_data[i].type == CostTypes::VelocityTracking) {
                fo_vel_idx_ = i;
            }
            if (settings_.cost_data[i].type == CostTypes::TorqueReg) {
                fo_tau_idx_ = i;
            }
            if (settings_.cost_data[i].type == CostTypes::ForceReg) {
                fo_force_idx_ = i;
            }
            if (settings_.cost_data[i].type == CostTypes::ForwardKinematics) {
                frame_tracking_idx_ = i;
            }
            if (settings_.cost_data[i].type == CostTypes::CentroidalConfiguration) {
                cent_config_idx_ = i;
            }
            if (settings_.cost_data[i].type == CostTypes::CentroidalVelocity) {
                cent_vel_idx_ = i;
            }
            if (settings_.cost_data[i].type == CostTypes::CentroidalForce) {
                cent_force_idx_ = i;
            }
        }

        if (settings_.log) {
            log_file_.open("hpipm_mpc_log.csv");
        }
    }

    HpipmMpc::~HpipmMpc() {
        if (settings_.log) {
            log_file_.close();
        }
    }

    void HpipmMpc::SetDynamicsConstraints(DynamicsConstraint constraints) {
        dynamics_constraint_ = std::make_unique<DynamicsConstraint>(std::move(constraints));
    }

    void HpipmMpc::SetCentroidalDynamicsConstraints(CentroidalDynamicsConstraint constraint) {
        centroidal_dynamics_constraint_ = std::make_unique<CentroidalDynamicsConstraint>(std::move(constraint));
    }

    void HpipmMpc::SetSrbConstraint(SRBConstraint constraint) {
        srb_constraint_ = std::make_unique<torc::mpc::SRBConstraint>(std::move(constraint));
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

    void HpipmMpc::SetForceBox(const BoxConstraint &stance_box, const BoxConstraint &swing_box) {
        stance_force_box_ = std::make_unique<BoxConstraint>(stance_box);
        swing_force_box_ = std::make_unique<BoxConstraint>(swing_box);

        if (stance_force_box_->GetFirstNode() != swing_force_box_->GetFirstNode()) {
            throw std::runtime_error("Force box constraints don't have the same first node!");
        }

        if (stance_force_box_->GetLastNode() != swing_force_box_->GetLastNode()) {
            throw std::runtime_error("Force box constraints don't have the same last node!");
        }

        if (stance_force_box_->GetNumConstraints() != swing_force_box_->GetNumConstraints()) {
            throw std::runtime_error("Force box constraints don't have the same number of constraints!");
        }
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

    void HpipmMpc::SetCollisionConstraint(CollisionConstraint constraints) {
        collision_ = std::make_unique<CollisionConstraint>(std::move(constraints));
    }

    void HpipmMpc::SetPolytopeConstraint(PolytopeConstraint constraint) {
        polytope_ = std::make_unique<PolytopeConstraint>(std::move(constraint));
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

    void HpipmMpc::SetFowardKinematicsCost(ForwardKinematicsCost cost) {
        fk_cost_ = std::make_unique<ForwardKinematicsCost>(std::move(cost));
    }

    void HpipmMpc::UpdateSetttings(MpcSettings settings) {
        settings_ = std::move(settings);
    }

    void HpipmMpc::CreateConstraints(int start_node, int end_node) {
        if (srb_constraint_ && centroidal_dynamics_constraint_) {
            throw std::runtime_error("Can't have both the SRB and centroidal constraint!");
        }

        // Maybe the first time I should set all the sizes for hpipm
        if (first_constraint_gen_) {
            if (!config_box_ || !vel_box_ || !tau_box_ || !friction_cone_ || !swing_constraint_ || !holonomic_
                || !dynamics_constraint_ || !stance_force_box_ || !swing_force_box_) {
                throw std::runtime_error("[HpipmMpc] a required constraint was not added!");
            }
            SetSizes();
            first_constraint_gen_ = false;
        }

        for (int node = start_node; node < end_node; node++) {
            // std::cerr << "node: " << node << std::endl;

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
                if (node < dynamics_constraint_->GetLastNode() - 1 && node < settings_.nodes - 1) {
                    // std::cerr << "Adding FO dynamics..." << std::endl;
                    dynamics_constraint_->GetLinDynamics(
                        traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1),
                        traj_.GetVelocity(node), traj_.GetVelocity(node + 1),
                        traj_.GetTau(node), force, settings_.dt[node], false,
                        qp[node].A, qp[node].B, qp[node].b);
                } else if (node == dynamics_constraint_->GetLastNode() - 1) {
                    // std::cerr << "Adding boundary dynamics..." << std::endl;
                    dynamics_constraint_->GetLinDynamics(
                        traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1),
                        traj_.GetVelocity(node), traj_.GetVelocity(node + 1),
                        traj_.GetTau(node), force, settings_.dt[node], true,
                        qp[node].A, qp[node].B, qp[node].b);

                } else if (node < settings_.nodes - 1) {
                    // std::cerr << "Adding ROM dynamics..." << std::endl;
                    if (centroidal_dynamics_constraint_) {
                        centroidal_dynamics_constraint_->GetLinDynamics(
                            traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1),
                            traj_.GetVelocity(node), traj_.GetVelocity(node + 1), force, settings_.dt[node],
                            qp[node].A, qp[node].B, qp[node].b);
                    }

                    if (srb_constraint_) {
                        // SRB model
                        srb_constraint_->GetLinDynamics(traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1),
                            traj_.GetVelocity(node), traj_.GetVelocity(node + 1), force, settings_.dt[node],
                            qp[node].A, qp[node].B, qp[node].b);
                    }
                }
            }

            // Config box constraints
            if (config_box_->IsInNodeRange(node)) { //(node >= config_box_->GetFirstNode() && node < config_box_->GetLastNode() + 1) && node != boundary_node_) {
                // std::cerr << "Adding config box..." << std::endl;
                // Set box indexes
                const auto& idxs = config_box_->GetIdxs();

                for (int i = 0; i < idxs.size(); i++) {
                    qp[node].idxbx[i] = idxs[i];
                }
                box_x_idx += idxs.size();

                // Bounds
                qp[node].lbx.head(idxs.size()) = config_box_->GetLowerBound(traj_.GetConfiguration(node).tail(nq_ - FLOATING_BASE));
                qp[node].ubx.head(idxs.size()) = config_box_->GetUpperBound(traj_.GetConfiguration(node).tail(nq_ - FLOATING_BASE));

            }

            // Vel box constraints
            if (vel_box_->IsInNodeRange(node)) { //(node >= vel_box_->GetFirstNode() && node < vel_box_->GetLastNode() + 1) && node != boundary_node_) {
                // std::cerr << "Adding vel box..." << std::endl;
                // Full order model this is in the state
                if (dynamics_constraint_->IsInNodeRange(node)) {
                    // Set box indexes
                    const auto& idxs = vel_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbx[i + box_x_idx] = idxs[i] + nv_;
                    }

                    // Bounds
                    qp[node].lbx.tail(idxs.size()) = vel_box_->GetLowerBound(traj_.GetVelocity(node).tail(nv_ - FLOATING_VEL));
                    qp[node].ubx.tail(idxs.size()) = vel_box_->GetUpperBound(traj_.GetVelocity(node).tail(nv_ - FLOATING_VEL));

                } else if (node < settings_.nodes - 1) {    // Reduced order model this is in the input
                    // Set box indexes
                    const auto& idxs = vel_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbu[i] = idxs[i] - FLOATING_VEL;
                    }

                    box_u_idx += idxs.size();

                    // Bounds
                    qp[node].lbu.head(idxs.size()) = vel_box_->GetLowerBound(traj_.GetVelocity(node).tail(nv_ - FLOATING_VEL));
                    qp[node].ubu.head(idxs.size()) = vel_box_->GetUpperBound(traj_.GetVelocity(node).tail(nv_ - FLOATING_VEL));
                }
            }

            // Torque box constraints
            if (dynamics_constraint_->IsInNodeRange(node) && tau_box_->IsInNodeRange(node)) { //(node >= tau_box_->GetFirstNode() && node < tau_box_->GetLastNode() + 1) && node != boundary_node_) {
                // std::cerr << "Adding tau box..." << std::endl;
                // Set box indexes
                const auto& idxs = tau_box_->GetIdxs();
                for (int i = 0; i < idxs.size(); i++) {
                    qp[node].idxbu[i] = idxs[i];
                }

                // Bounds
                qp[node].lbu.head(idxs.size()) = tau_box_->GetLowerBound(traj_.GetTau(node));
                qp[node].ubu.head(idxs.size()) = tau_box_->GetUpperBound(traj_.GetTau(node));
            }

            // Force box constraints
            for (int frame_idx = 0; frame_idx < settings_.num_contact_locations; frame_idx++) {
                const std::string frame = settings_.contact_frames[frame_idx];
                if (in_contact_[frame][node] && stance_force_box_->IsInNodeRange(node) && node < settings_.nodes - 1) {
                    // Set box indexes
                    const auto& idxs = stance_force_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbu[i + ntau_ + 3*frame_idx] = idxs[i] + 3*frame_idx + ntau_;
                    }

                    // Bounds
                    qp[node].lbu.segment<3>(ntau_ + 3*frame_idx) = stance_force_box_->GetLowerBound(traj_.GetForce(node, frame));
                    qp[node].ubu.segment<3>(ntau_ + 3*frame_idx) = stance_force_box_->GetUpperBound(traj_.GetForce(node, frame));

                } else if (!in_contact_[frame][node] && swing_force_box_->IsInNodeRange(node) && node < settings_.nodes - 1) {
                    // Set box indexes
                    const auto& idxs = swing_force_box_->GetIdxs();
                    for (int i = 0; i < idxs.size(); i++) {
                        qp[node].idxbu[i + ntau_ + 3*frame_idx] = idxs[i] + 3*frame_idx + ntau_;
                    }

                    // Bounds
                    qp[node].lbu.segment<3>(ntau_ + 3*frame_idx) = swing_force_box_->GetLowerBound(traj_.GetForce(node, frame));
                    qp[node].ubu.segment<3>(ntau_ + 3*frame_idx) = swing_force_box_->GetUpperBound(traj_.GetForce(node, frame));
                }
            }

            // Friction cone constraints
            if (friction_cone_->IsInNodeRange(node) && node < settings_.nodes - 1) {
                // std::cerr << "Adding friction cone..." << std::endl;
                const int nconstraints = friction_cone_->GetNumConstraints();
                int col = ntau_;
                for (int contact = 0; contact < settings_.num_contact_locations; contact++) {
                    const auto [d_block, lin_seg]
                        = friction_cone_->GetLinearization(force.segment<3>(3*contact));
                    qp[node].D.block(ineq_row_idx, col, nconstraints, CONTACT_3DOF) =
                        in_contact_[settings_.contact_frames[contact]][node]*d_block;

                    qp[node].lg.segment(ineq_row_idx, lin_seg.size()) = -in_contact_[settings_.contact_frames[contact]][node]*lin_seg;
                    qp[node].ug.segment(ineq_row_idx, lin_seg.size()) = -in_contact_[settings_.contact_frames[contact]][node]*lin_seg;

                    for (int i = 0; i < 4; i++) {
                        qp[node].ug_mask(ineq_row_idx + i) = 0;
                    }
                    col += CONTACT_3DOF;
                    ineq_row_idx += nconstraints;
                }
            }

            // Swing height
            if (swing_constraint_->IsInNodeRange(node)) {
                // std::cerr << "Adding swing..." << std::endl;
                for (const auto& frame : settings_.contact_frames) {
                    const auto [c_block, y_segment] =
                        swing_constraint_->GetLinearization(traj_.GetConfiguration(node), swing_traj_[frame][node], frame);
                    qp[node].C.block(ineq_row_idx, 0, y_segment.size(), nv_) = c_block;
                    // TODO: Consider removing the buffer
                    qp[node].lg(ineq_row_idx) = -y_segment(0) - settings_.swing_buffer;  // Buffer
                    qp[node].ug(ineq_row_idx) = -y_segment(0) + settings_.swing_buffer;  // Buffer
                    ineq_row_idx += y_segment.size();
                }
            }

            // Holonomic
            if (holonomic_->IsInNodeRange(node) && node < settings_.nodes - 1) {
                // std::cerr << "Adding holonomic..." << std::endl;
                for (const auto& frame : settings_.contact_frames) {
                    const auto [jac, y_segment] =
                        holonomic_->GetLinearization(traj_.GetConfiguration(node), traj_.GetVelocity(node), frame);
                    if (dynamics_constraint_->IsInNodeRange(node)) {
                        qp[node].C.middleRows(ineq_row_idx, y_segment.size()) = in_contact_[frame][node]*jac;
                    } else {
                        // The issue seemed to go away when I used the LOCAL frame for the constraint
                        qp[node].C.middleRows(ineq_row_idx, y_segment.size()) =
                            in_contact_[frame][node]*jac.leftCols(nv_ + FLOATING_VEL);
                        qp[node].D.block(ineq_row_idx, 0, y_segment.size(), nv_ - FLOATING_VEL) =
                            in_contact_[frame][node]*jac.rightCols(nv_ - FLOATING_VEL);
                    }
                    qp[node].lg.segment<2>(ineq_row_idx) = -in_contact_[frame][node]*y_segment;
                    qp[node].ug.segment<2>(ineq_row_idx) = -in_contact_[frame][node]*y_segment;

                    ineq_row_idx += y_segment.size();
                }
            }

            // Collision
            if (collision_->IsInNodeRange(node)) {
                for (int i = 0; i < collision_->GetNumCollisions(); i++) {
                    const auto [c_block, y_segment] =
                        collision_->GetLinearization(traj_.GetConfiguration(node), i);
                    qp[node].C.block(ineq_row_idx, 0, y_segment.size(), nv_) = c_block;
                    qp[node].lg.segment(ineq_row_idx, 1) = -y_segment;
                    qp[node].ug_mask(ineq_row_idx) = 0;
                    ineq_row_idx++;
                }
            }

            // Polytope
            if (polytope_->IsInNodeRange(node)) {
                // std::cerr << "Adding polytope..." << std::endl;
                int slack_idx = 0;
                for (const auto& frame : settings_.contact_frames) {
                    matrixx_t jac;
                    vectorx_t ub, lb;
                    polytope_->GetLinearization(traj_.GetConfiguration(node),
                        contact_info_[frame][node], frame, jac, ub, lb);
                    // std::cout << "frame: " << frame << std::endl;

                    // std::cout << "node: " << node << " in contact " << in_contact_[frame][node] << std::endl;
                    // std::cout << "polytope A:\n" << contact_info_[frame][node].A_ << std::endl;
                    // std::cout << "polytope b: " << contact_info_[frame][node].b_.transpose() << std::endl;

                    qp[node].C.block(ineq_row_idx, 0, PolytopeConstraint::POLYTOPE_SIZE/2, nv_)
                        = jac;

                    qp[node].lg.segment<PolytopeConstraint::POLYTOPE_SIZE/2>(ineq_row_idx) = lb;
                    qp[node].ug.segment<PolytopeConstraint::POLYTOPE_SIZE/2>(ineq_row_idx) = ub;

                    ineq_row_idx += PolytopeConstraint::POLYTOPE_SIZE/2;
                }
            }

            // std::cout << "node: " << node << std::endl;
            // for (const auto& frame : settings_.contact_frames) {
            //     std::cout << "[" << frame << "] in contact: " << in_contact_[frame][node]
            //         << ", swing height: " << swing_traj_[frame][node] << std::endl;
            // }
            // std::cout << "idxbx: ";
            // for (const auto& idx : qp[node].idxbx) {
            //     std::cout << idx << ", ";
            // }
            // std::cout << std::endl;
            // std::cout << "lbx: " << qp[node].lbx.transpose() << std::endl;
            // std::cout << "ubx: " << qp[node].ubx.transpose() << std::endl;
            //
            // std::cout << "idxbu: ";
            // for (const auto& idx : qp[node].idxbu) {
            //     std::cout << idx << ", ";
            // }
            // std::cout << std::endl;
            // std::cout << "lbu: " << qp[node].lbu.transpose() << std::endl;
            // std::cout << "ubu: " << qp[node].ubu.transpose() << std::endl;
            //
            // std::cout << "D:\n" << qp[node].D << std::endl;
            // std::cout << "C:\n" << qp[node].C << std::endl;
            // std::cout << "lg: " << qp[node].lg.transpose() << std::endl;
            // std::cout << "ug: " << qp[node].ug.transpose() << std::endl;
            // std::cout << "ugmask: " << qp[node].ug_mask.transpose() << std::endl;
            //
            // std::cout << "A:\n" << qp[node].A << std::endl;
            // std::cout << "B:\n" << qp[node].B << std::endl;
            // std::cout << "b: " << qp[node].b.transpose() << std::endl;
        }
    }

    void HpipmMpc::CreateCost(int start_node, int end_node) {
        if (!vel_tracking_ || !tau_tracking_ || !force_tracking_) {
            throw std::runtime_error("[HpipmMpc] Required cost not set!");
        }

        for (int node = start_node; node < end_node - 1; node++) {

            if (vel_tracking_->IsInNodeRange(node)) {
                const auto [hess, lin]
                    = vel_tracking_->GetQuadraticApprox(traj_.GetVelocity(node), GetVelocityTarget(node),
                        dynamics_constraint_->IsInNodeRange(node) ?
                            settings_.cost_data[fo_vel_idx_].weight : settings_.cost_data[cent_vel_idx_].weight);

                if (dynamics_constraint_->IsInNodeRange(node)) {
                    qp[node].Q.bottomRightCorner(nv_, nv_) = hess;
                    qp[node].q.tail(nv_) = lin;
                } else {
                    qp[node].Q.bottomRightCorner<FLOATING_VEL, FLOATING_VEL>() = hess.topLeftCorner<FLOATING_VEL, FLOATING_VEL>();
                    qp[node].q.tail<FLOATING_VEL>() = lin.head<FLOATING_VEL>();

                    qp[node].R.topLeftCorner(ntau_, ntau_) = hess.bottomRightCorner(ntau_, ntau_);
                    qp[node].r.head(ntau_) = lin.tail(ntau_);
                }
            }

            if (tau_tracking_->IsInNodeRange(node) && dynamics_constraint_->IsInNodeRange(node)) {
                const auto [hess, lin]
                    = tau_tracking_->GetQuadraticApprox(traj_.GetTau(node), GetTauTarget(node),
                        settings_.cost_data[fo_tau_idx_].weight);

                qp[node].R.topLeftCorner(ntau_, ntau_) = hess;
                qp[node].r.head(ntau_) = lin;
            }

            if (force_tracking_->IsInNodeRange(node)) {
                int block_idx = ntau_;
                for (const auto& frame : settings_.contact_frames) {
                    const auto [hess, lin]
                        = force_tracking_->GetQuadraticApprox(
                            traj_.GetForce(node, frame), GetForceTarget(node, frame),
                            dynamics_constraint_->IsInNodeRange(node) ?
                            settings_.cost_data[fo_force_idx_].weight : settings_.cost_data[cent_force_idx_].weight);
                    qp[node].R.block<CONTACT_3DOF, CONTACT_3DOF>(block_idx, block_idx) = hess;
                    qp[node].r.segment<CONTACT_3DOF>(block_idx) = lin;
                    block_idx += CONTACT_3DOF;
                }
            }

            if (config_tracking_->IsInNodeRange(node)) {
                const auto [hess, lin] =
                    config_tracking_->GetQuadraticApprox(traj_.GetConfiguration(node), GetConfigTarget(node),
                    dynamics_constraint_->IsInNodeRange(node) ?
                        settings_.cost_data[fo_config_idx_].weight : settings_.cost_data[cent_config_idx_].weight);
                qp[node].Q.topLeftCorner(nv_, nv_) = hess;
                qp[node].q.head(nv_) = lin;
            }

            if (fk_cost_ && fk_cost_->IsInNodeRange(node)) {
                for (const auto& frame : settings_.contact_frames) {
                    const auto [hess, lin] = fk_cost_->GetQuadraticApprox(
                        traj_.GetConfiguration(node),
                        GetEndEffectorTarget(node, frame),
                        settings_.cost_data[frame_tracking_idx_].weight, frame);
                    qp[node].Q.topLeftCorner(nv_, nv_) += hess;
                    qp[node].q.head(nv_) += lin;
                }
            }

            qp[node].Q *= settings_.dt[node];
            qp[node].q *= settings_.dt[node];
            qp[node].R *= settings_.dt[node];
            qp[node].r *= settings_.dt[node];
            qp[node].S *= settings_.dt[node];

            // std::cout << "node: " << node << std::endl;
            // std::cout << "Q:\n" << qp[node].Q << std::endl;
            // std::cout << "q:\n" << qp[node].q.transpose() << std::endl;
            // std::cout << "R:\n" << qp[node].R << std::endl;
            // std::cout << "r:\n" << qp[node].r.transpose() << std::endl;
        }

        if (end_node == settings_.nodes) {
            // Terminal cost
            int node = settings_.nodes - 1;
            if (dynamics_constraint_->IsInNodeRange(node)) {
                const auto [hess, lin]
                    = vel_tracking_->GetQuadraticApprox(traj_.GetVelocity(node), GetVelocityTarget(node),
                            dynamics_constraint_->IsInNodeRange(node) ?
                            settings_.cost_data[fo_vel_idx_].weight : settings_.cost_data[cent_vel_idx_].weight);
                qp[node].Q.bottomRightCorner<FLOATING_VEL, FLOATING_VEL>() = hess.topLeftCorner<FLOATING_VEL, FLOATING_VEL>();
                qp[node].q.tail<FLOATING_VEL>() = lin.head<FLOATING_VEL>();
            }

            const auto [hessq, linq] =
                        config_tracking_->GetQuadraticApprox(traj_.GetConfiguration(node), GetConfigTarget(node),
                        dynamics_constraint_->IsInNodeRange(node) ?
                        settings_.cost_data[fo_config_idx_].weight : settings_.cost_data[cent_config_idx_].weight);
            qp[node].Q.topLeftCorner(nv_, nv_) = hessq;
            qp[node].q.head(nv_) = linq;

            for (const auto& frame : settings_.contact_frames) {
                const auto [hess, lin] = fk_cost_->GetQuadraticApprox(
                    traj_.GetConfiguration(node),
                    GetEndEffectorTarget(node, frame),
                    settings_.cost_data[frame_tracking_idx_].weight, frame);
                qp[node].Q.topLeftCorner(nv_, nv_) += hess;
                qp[node].q.head(nv_) += lin;
            }

            qp[node].Q *= settings_.terminal_weight*settings_.dt[node];
            qp[node].q *= settings_.terminal_weight*settings_.dt[node];
        }
    }


    void HpipmMpc::SetSizes() {
        // Resize all the QP mats and set them to zero.
        for (int node = 0; node < settings_.nodes; node++) {
            int nx1, nx2;
            int nu = ntau_ + CONTACT_3DOF*settings_.num_contact_locations;
            int nx_box = 0;
            int nu_box = 0;
            int n_other_constraints = 0;
            int n_slack = 0;
            if (friction_cone_->IsInNodeRange(node) && node < settings_.nodes - 1) {    // No inputs in the last node
                n_other_constraints += settings_.num_contact_locations*friction_cone_->GetNumConstraints();
            }
            if (swing_constraint_->IsInNodeRange(node)) {
                n_other_constraints += swing_constraint_->GetNumConstraints();
            }
            if (holonomic_->IsInNodeRange(node) && node < settings_.nodes - 1) {    // Assuming the last node is always ROM.
                n_other_constraints += holonomic_->GetNumConstraints();
            }
            if (collision_->IsInNodeRange(node)) {
                n_other_constraints += collision_->GetNumConstraints();
            }
            if (polytope_->IsInNodeRange(node)) {
                n_other_constraints += polytope_->GetNumConstraints();
                n_slack += polytope_->GetNumConstraints();
            }

            if (node < dynamics_constraint_->GetLastNode() - 1) {
                nx1 = nv_ + nv_;
                nx2 = nx1;
                if (config_box_->IsInNodeRange(node)) {
                    nx_box += nv_ - FLOATING_VEL;
                }
                if (vel_box_->IsInNodeRange(node)) {
                    nx_box += nv_ - FLOATING_VEL;
                }
                if (tau_box_->IsInNodeRange(node)) {
                    nu_box += ntau_;
                }
                if (swing_force_box_->IsInNodeRange(node)) {
                    nu_box += settings_.num_contact_locations*swing_force_box_->GetNumConstraints();
                }
            } else if (node == dynamics_constraint_->GetLastNode() - 1) {
                nx1 = nv_ + nv_;
                nx2 = nv_ + FLOATING_VEL;
                if (config_box_->IsInNodeRange(node)) {
                    nx_box += nv_ - FLOATING_VEL;
                }
                if (vel_box_->IsInNodeRange(node)) {
                    nx_box += nv_ - FLOATING_VEL;
                }
                if (tau_box_->IsInNodeRange(node)) {
                    nu_box += ntau_;
                }
                if (swing_force_box_->IsInNodeRange(node)) {
                    nu_box += settings_.num_contact_locations*swing_force_box_->GetNumConstraints();
                }
            } else {
                nx1 = nv_ + FLOATING_VEL;
                nx2 = nx1;
                if (config_box_->IsInNodeRange(node)) {
                    nx_box += nv_ - FLOATING_VEL;
                }
                if (vel_box_->IsInNodeRange(node)) {
                    nu_box += ntau_;
                }
                if (swing_force_box_->IsInNodeRange(node)) {
                    nu_box += settings_.num_contact_locations*swing_force_box_->GetNumConstraints();
                }
            }

            // No inputs in the last node
            if (node == settings_.nodes - 1) {
                nu = 0;
                nu_box = 0;
            }

            // std::cout << "node: " << node << ", nx1: " << nx1 << ", nx2: " << nx2 << ", nu: " << nu << std::endl;

            if (node < settings_.nodes - 1) {
                // Dynamics
                qp[node].A = matrixx_t::Zero(nx2, nx1);
                qp[node].B = matrixx_t::Zero(nx2, nu);
                qp[node].b = vectorx_t::Zero(nx2);
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
            qp[node].ug_mask = vectorx_t::Ones(n_other_constraints);

            solution_[node].x = vectorx_t::Zero(nx1);
            solution_[node].u = vectorx_t::Zero(nu);
        }

        solver_ = std::make_unique<hpipm::OcpQpIpmSolver>(qp, qp_settings);
    }

    void HpipmMpc::NanCheck() {
        auto print_nan_mat = [](const matrixx_t& mat, int node, const std::string& name)->bool {
            bool is_nan = false;
            for (int row = 0; row < mat.rows(); row++) {
                for (int col = 0; col < mat.cols(); col++) {
                    if (std::isnan(mat(row, col))) {
                        is_nan = true;
                    }
                }
            }
            if (is_nan) {
                std::cerr << "Node: " << node << " " + name + ":\n" << mat << std::endl;
                throw std::runtime_error("NaN detected in a matrix!");
            }
            return is_nan;
        };

        auto print_nan_vec = [](const vectorx_t& vec, int node, const std::string& name)->bool {
            bool is_nan = false;
            if (vec.data()) {
                for (int row = 0; row < vec.size(); row++) {
                    if (std::isnan(vec[row])) {
                        std::cerr << "NaN detected at (" << row << ") during node " << node << std::endl;
                        is_nan = true;
                    }
                }
            }

            if (is_nan) {
                std::cerr << name + ":\n" << vec << std::endl;
                throw std::runtime_error("NaN detected in a vector!");
            }
            return is_nan;
        };


        for (int node = 0; node < settings_.nodes; node++) {
            if (node < settings_.nodes - 1) {
                print_nan_mat(qp[node].A, node, "A");

                print_nan_mat(qp[node].B, node, "B");

                print_nan_vec(qp[node].b, node, "b");
            }


            print_nan_mat(qp[node].Q, node, "Q");

            print_nan_mat(qp[node].R, node, "R");

            print_nan_mat(qp[node].S, node, "S");

            print_nan_vec(qp[node].q, node, "q");

            print_nan_vec(qp[node].r, node, "r");

            print_nan_vec(qp[node].lbx, node, "lbx");

            print_nan_vec(qp[node].ubx, node, "ubx");

            print_nan_vec(qp[node].lbu, node, "lbu");

            print_nan_vec(qp[node].ubu, node, "ubu");

            print_nan_vec(qp[node].lg, node, "lg");

            print_nan_vec(qp[node].ug, node, "ug");
        }
    }

    void HpipmMpc::CreateQPData() {
        // Create the constraints
        constraint_timer_.Tic();
        CreateConstraints(1, settings_.nodes);
        constraint_timer_.Toc();
        if (settings_.verbose) {
            std::cout << "constraint time: " << constraint_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl;
        }
        // std::cout << "constraint time: " << constraint_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl;

        // Create the cost
        cost_timer_.Tic();
        CreateCost(1, settings_.nodes);
        cost_timer_.Toc();
        if (settings_.verbose) {
            std::cout << "cost time: " << cost_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl;
        }
        // std::cout << "cost time: " << cost_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl;
    }

    void HpipmMpc::CreateNode0Data() {
        CreateConstraints(0, 1);

        CreateCost(0, 1);
    }

    hpipm::HpipmStatus HpipmMpc::Compute(double time, const vectorx_t &q0, const vectorx_t &v0, Trajectory& traj_out) {
        if (std::abs(q0.segment<4>(3).norm() - 1) > 1e-8) {
            std::cerr << "q: " << q0.transpose() << std::endl;
            throw std::runtime_error("Initial condition does not have a normalized quaternion!");
        }

        traj_.SetConfiguration(0, q0);
        traj_.SetVelocity(0, v0);

        // Linearize node 0 (initial condition)
        CreateNode0Data();

        // NanCheck();

        solve_timer_.Tic();
        result_ = solver_->solve(vectorx_t::Zero(model_.GetVelDim() + model_.GetVelDim()),
            qp, solution_);
        solve_timer_.Toc();

        assert(solution_[0].x == vectorx_t::Zero(2*nv_));   // Verify the initial condition is correct

        statistics_ = solver_->getSolverStatistics();

        if (settings_.verbose) {
            std::cout << "Res: " << result_ << std::endl;
            std::cout << statistics_ << std::endl;
            std::cout << "solve time: " << solve_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl;
            std::cout << "Constraint violation: " << GetConstraintViolation(solution_, 1) << std::endl;
            std::cout << "Cost: " << GetCost(solution_, 1) << std::endl;
        }
        // std::cout << "solve time: " << solve_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl;


        // Line Search
        ls_timer_.Tic();
        // TODO: Put back if I want
        // std::tie(prev_violation_, prev_cost_) = LineSearch(solution_);
        prev_violation_ = -1;
        prev_cost_ = -1;
        ls_timer_.Toc();

        if (settings_.verbose) {
            std::cout << "Post LS constraint violation: " << prev_violation_ << std::endl;
            std::cout << "Post LS cost: " << prev_cost_ << std::endl;
            std::cout << "line search time: " << ls_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << "ms" << std::endl;
        }

        ConvertQpSolToTraj(alpha_);
        traj_out = traj_;

        if (settings_.verbose) {
            PrintNodeInfo();
        }

        solve_counter_++;

        return result_;
    }

    void HpipmMpc::ConvertQpSolToTraj(double alpha) {
        for (int node = 0; node < settings_.nodes; node++) {
            // std::cout << "node: " << node << std::endl;
            // std::cout << "traj idx: " << traj_idx << std::endl;
            // std::cout << "x: " << solution_[node].x.transpose() << std::endl;
            // std::cout << "u: " << solution_[node].u.transpose() << std::endl;

            if (dynamics_constraint_->IsInNodeRange(node)) {
                if (node != 0) {
                    traj_.SetConfiguration(node, models::ConvertdqToq<double>(alpha*solution_[node].x.head(nv_),
                        traj_.GetConfiguration(node)));

                    traj_.SetVelocity(node, traj_.GetVelocity(node) + alpha*solution_[node].x.tail(nv_));
                }
                if (node < settings_.nodes - 1) {
                    traj_.SetTau(node, traj_.GetTau(node) + alpha*solution_[node].u.head(ntau_));
                }
            } else {
                traj_.SetConfiguration(node, models::ConvertdqToq<double>(alpha*solution_[node].x.head(nv_),
                    traj_.GetConfiguration(node)));

                vectorx_t v = vectorx_t::Zero(nv_);
                v.head<FLOATING_VEL>() = alpha*solution_[node].x.tail<FLOATING_VEL>();

                if (node < settings_.nodes - 1) {
                    v.tail(ntau_) = alpha*solution_[node].u.head(ntau_);
                }
                traj_.SetVelocity(node, traj_.GetVelocity(node) + v);
            }

            if (node < settings_.nodes - 1) {
                for (int i = 0; i < settings_.contact_frames.size(); i++) {
                    traj_.SetForce(node, settings_.contact_frames[i],
                        traj_.GetForce(node, settings_.contact_frames[i]) + alpha*solution_[node].u.segment<CONTACT_3DOF>(ntau_ + i*3));
                }
            }

            if (std::abs(traj_.GetConfiguration(node).segment<4>(3).norm() - 1) > 1e-8) {
                std::cerr << "q: " << traj_.GetConfiguration(node) << std::endl;
                throw std::runtime_error("Output quaternion is not normalized!");
            }
        }
    }

    // --------- Get Targets --------- //
    vectorx_t HpipmMpc::GetVelocityTarget(int node) const {
        return v_target_[node];
    }

    vectorx_t HpipmMpc::GetTauTarget(int node) const {
        return tau_target_[node];
    }

    vector3_t HpipmMpc::GetForceTarget(int node, const std::string& frame) const {
        if (in_contact_.at(frame)[node]) {
            int num_in_contact = 0;
            for (int j = 0; j < settings_.num_contact_locations; j++) {
                if (in_contact_.at(settings_.contact_frames.at(j))[node]) {
                    num_in_contact++;
                }
            }
            vector3_t force_out;
            force_out << 0, 0, model_.GetMass()*9.81/static_cast<double>(num_in_contact);
            return force_out;
        }

        return vector3_t::Zero();
    }

    vectorx_t HpipmMpc::GetConfigTarget(int node) const {
        return q_target_[node];
    }

    vector3_t HpipmMpc::GetEndEffectorTarget(int node, const std::string &frame) const {
        return end_effector_targets_.at(frame)[node];
    }

    // --------- Set Targets --------- //
    void HpipmMpc::SetVelTarget(const SimpleTrajectory &v_target) {
        v_target_ = v_target;
    }

    void HpipmMpc::SetConfigTarget(const SimpleTrajectory &q_target) {
        q_target_ = q_target;
    }

    void HpipmMpc::SetForwardKinematicsTarget(const std::map<std::string, std::vector<vector3_t> > &fk_positions) {
        end_effector_targets_ = fk_positions;
    }


    void HpipmMpc::SetLinTraj(const Trajectory &traj_in) {
        traj_ = traj_in;
    }


    void HpipmMpc::SetLinTrajConfig(const SimpleTrajectory &config_traj) {
        if (config_traj.GetNumNodes() != traj_.GetNumNodes()) {
            throw std::runtime_error("[SetLinTrajConfig] Invalid config trajectory size!");
        }

        for (int i = 0; i < traj_.GetNumNodes(); i++) {
            traj_.SetConfiguration(i, config_traj.GetNodeData(i));
        }
    }

    void HpipmMpc::SetLinTrajVel(const SimpleTrajectory &vel_traj) {
        if (vel_traj.GetNumNodes() != traj_.GetNumNodes()) {
            throw std::runtime_error("[SetLinTrajConfig] Invalid config trajectory size!");
        }

        for (int i = 0; i < traj_.GetNumNodes(); i++) {
            traj_.SetVelocity(i, vel_traj.GetNodeData(i));
        }
    }



    void HpipmMpc::UpdateContactSchedule(const ContactSchedule &sched) {
        vector4_t polytope_delta;
        polytope_delta << settings_.polytope_delta, settings_.polytope_delta,
            -settings_.polytope_delta, -settings_.polytope_delta;

        vector4_t polytope_convergence_scalar;
        polytope_convergence_scalar << 1, 1, -1, -1;

        for (const auto& [frame, schedule] : sched.GetScheduleMap()) {
            if (in_contact_.contains(frame)) {

                const auto& polytopes = sched.GetPolytopes(frame);

                int current_contact = -1;
                if (sched.InContact(frame, 0)) {
                    current_contact = sched.GetContactIndex(frame, 0);
                }

                // Break the continuous time contact schedule into each node
                double time = 0;
                int contact_idx = 0;
                for (int node = 0; node < settings_.nodes; node++) {
                    if (sched.InContact(frame, time)) {
                        in_contact_[frame][node] = 1;
                        traj_.SetInContact(node, frame, true);

                        contact_idx = sched.GetContactIndex(frame, time);

                        contact_info_[frame][node] = polytopes[contact_idx];
                        if (contact_idx != current_contact) {
                            // Only add the margin for future polytopes
                            contact_info_[frame][node].b_ -= polytope_delta;
                        }
                    } else {
                        in_contact_[frame][node] = 0;
                        traj_.SetInContact(node, frame, false);

                        contact_info_[frame][node] = ContactSchedule::GetDefaultContactInfo();

                        // TODO: This seems to make it worse, might need to be careful with the function used for the scaling
                        // if (contact_idx + 1 < polytopes.size()) {
                        //     contact_info_[frame][node] = polytopes[contact_idx + 1];
                        //     contact_info_[frame][node].b_ = contact_info_[frame][node].b_  //- polytope_delta
                        //     + GetPolytopeConvergence(frame, time, sched)*polytope_convergence_scalar;
                        // } else {
                        //     contact_info_[frame][node] = ContactSchedule::GetDefaultContactInfo();
                        // }
                    }
                    time += settings_.dt[node];
                }
            } else {
                throw std::runtime_error("Contact schedule contains contact frames not recognized by the MPC: " + frame);
            }
        }

        int frame_idx = 0;
        for (auto& [frame, traj] : swing_traj_) {
            sched.CreateSwingTraj(frame, settings_.apex_height, settings_.default_ground_height,    // TODO: make the height adjustable
                settings_.apex_time, settings_.dt, traj);
            frame_idx++;
        }
    }

    double HpipmMpc::GetPolytopeConvergence(const std::string &frame, double time, const ContactSchedule& cs) const {
        // Compute time left until the next contact
        double swing_dur = cs.GetSwingDuration(frame, time);
        double start_time = cs.GetSwingStartTime(frame, time);
        double lambda = 1 - ((time - start_time) / swing_dur);

        if (lambda < 0 || lambda > 1) {
            throw std::runtime_error("[GetPolytopeConvergence] Invalid lambda!");
        }

        // Multiply range by a positive number
        return lambda*settings_.polytope_shrinking_rad;
    }

    double HpipmMpc::GetConstraintViolation(const std::vector<hpipm::OcpQpSolution>& sol, double alpha) {
        double violation = 0;
        for (int node = 0; node < settings_.nodes; node++) {
            vectorx_t force(CONTACT_3DOF*settings_.num_contact_locations);
            int force_idx = 0;
            for (const auto& frame : settings_.contact_frames) {
                force.segment<CONTACT_3DOF>(force_idx) = traj_.GetForce(node, frame);
                force_idx += CONTACT_3DOF;
            }

            vectorx_t dq = sol[node].x.head(nv_);
            vectorx_t dv = vectorx_t::Zero(nv_);
            vectorx_t dtau = vectorx_t::Zero(ntau_);
            dv.head<FLOATING_VEL>() = sol[node].x.segment<FLOATING_VEL>(nv_);
            if (dynamics_constraint_->IsInNodeRange(node)) {
                dv.tail(ntau_) = sol[node].x.segment(FLOATING_VEL + nv_, ntau_);
                if (node < settings_.nodes - 1) {
                    dtau = sol[node].u.head(ntau_);
                }
            } else if (node  < settings_.nodes - 1) {
                dv.tail(ntau_) = sol[node].u.head(ntau_);
            }
            vectorx_t df = vectorx_t::Zero(CONTACT_3DOF*settings_.num_contact_locations);
            if (node < settings_.nodes - 1) {
                df = sol[node].u.tail(CONTACT_3DOF*settings_.num_contact_locations);
            }

            dq *= alpha;
            dv *= alpha;
            dtau *= alpha;
            df *= alpha;

            if (dynamics_constraint_->IsInNodeRange(node) && node < settings_.nodes - 1) {

                vectorx_t dq2 = sol[node+1].x.head(nv_);
                vectorx_t dv2 = sol[node+1].x.tail(nv_);
                dq2 *= alpha;
                dv2 *= alpha;

                auto[int_vio, dyn_vio] =
                    dynamics_constraint_->GetViolation(traj_.GetConfiguration(node),
                    traj_.GetConfiguration(node + 1), traj_.GetVelocity(node),
                    traj_.GetVelocity(node + 1), traj_.GetTau(node), force, settings_.dt[node],
                    dq, dq2, dv, dv2, dtau, df);

                if (node == dynamics_constraint_->GetLastNode() - 1) {
                    // Boundary dynamics
                    dyn_vio.tail(nv_ - FLOATING_VEL).setZero();
                }

                // std::cout << "Dynamics vio: |" << dyn_vio.squaredNorm() << "| " << dyn_vio.transpose() << std::endl;
                // std::cout << "Integration vio: |" << int_vio.squaredNorm() << "| " << int_vio.transpose() << std::endl;
                // std::cout << "v: " << traj_.GetVelocity(node).transpose() << std::endl;
                // std::cout << "dv: " << dv.transpose() << std::endl;
                // std::cout << "v2: " << traj_.GetVelocity(node+1).transpose() << std::endl;
                // std::cout << "dv2: " << dv2.transpose() << std::endl;
                // std::cout << "dtau: " << dtau.transpose() << std::endl;
                // std::cout << "q: " << traj_.GetConfiguration(node).transpose() << std::endl;
                // std::cout << "dq: " << dq.transpose() << std::endl;
                // std::cout << "tau: " << traj_.GetTau(node).transpose() << std::endl;
                // std::cout << "dtau: " << dtau.transpose() << std::endl;
                // std::cout << "F: " << force.transpose() << std::endl;
                // std::cout << "dF: " << df.transpose() << std::endl;

                violation += dyn_vio.squaredNorm();
                violation += int_vio.squaredNorm();

            } else if (node < traj_.GetNumNodes() - 1) {
                vectorx_t dq2 = sol[node+1].x.head(nv_);
                vectorx_t dv2 = vectorx_t::Zero(nv_);
                dv2.head<FLOATING_VEL>() = sol[node+1].x.segment<FLOATING_VEL>(nv_);
                if (node < traj_.GetNumNodes() - 2) {
                    dv2.tail(ntau_) = sol[node+1].u.head(ntau_);
                }

                dq2 *= alpha;
                dv2 *= alpha;

                dtau.setZero();

                vectorx_t int_vio, dyn_vio;

                if (centroidal_dynamics_constraint_) {
                    std::tie(int_vio, dyn_vio) = centroidal_dynamics_constraint_->GetViolation(
                        traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1), traj_.GetVelocity(node),
                        traj_.GetVelocity(node + 1), force, settings_.dt[node],
                        dq, dq2, dv, dv2, df);
                }

                if (srb_constraint_) {
                    std::tie(int_vio, dyn_vio) = srb_constraint_->GetViolation(
                        traj_.GetConfiguration(node), traj_.GetConfiguration(node + 1), traj_.GetVelocity(node),
                        traj_.GetVelocity(node + 1), force, settings_.dt[node],
                        dq, dq2, dv, dv2, df);
                }

                // std::cout << "Dynamics vio: |" << dyn_vio.squaredNorm() << "| " << dyn_vio.transpose() << std::endl;
                // std::cout << "Integration vio: |" << int_vio.squaredNorm() << "| " << int_vio.transpose() << std::endl;

                violation += dyn_vio.squaredNorm();
                violation += int_vio.squaredNorm();
            }

            if (config_box_->IsInNodeRange(node)) {
                vectorx_t vio_vec = config_box_->GetViolation(traj_.GetConfiguration(node).tail(nv_ - FLOATING_VEL),
                    dq.tail(nv_ - FLOATING_VEL));
                // std::cout << "Config box vio: |" << vio_vec.squaredNorm() << "| " << vio_vec.transpose() << std::endl;
                violation += vio_vec.squaredNorm();
            }

            if (vel_box_->IsInNodeRange(node)) {
                vectorx_t vio_vec = vel_box_->GetViolation(traj_.GetVelocity(node).tail(nv_ - FLOATING_VEL),
                    dv.tail(nv_ - FLOATING_VEL));
                // std::cout << "Vel box vio: |" << vio_vec.squaredNorm() << "| " << vio_vec.transpose() << std::endl;
                violation += vio_vec.squaredNorm();
            }

            if (tau_box_->IsInNodeRange(node) && dynamics_constraint_->IsInNodeRange(node)) {
                vectorx_t vio_vec = tau_box_->GetViolation(traj_.GetTau(node), dtau);
                // std::cout << "Tau box vio: |" << vio_vec.squaredNorm() << "| " << vio_vec.transpose() << std::endl;
                violation += vio_vec.squaredNorm();
            }

            for (int i = 0; i < settings_.contact_frames.size(); i++) {
                const std::string frame = settings_.contact_frames[i];
                if (in_contact_[frame][node] && stance_force_box_->IsInNodeRange(node) && node < settings_.nodes - 1) {
                    vectorx_t vio_vec = stance_force_box_->GetViolation(traj_.GetForce(node, frame), df.segment<3>(3*i));
                    violation += vio_vec.squaredNorm();
                } else if (!in_contact_[frame][node] && swing_force_box_->IsInNodeRange(node) && node < settings_.nodes - 1) {
                    vectorx_t vio_vec = swing_force_box_->GetViolation(traj_.GetForce(node, frame), df.segment<3>(3*i));
                    violation += vio_vec.squaredNorm();
                }
            }

            if (friction_cone_->IsInNodeRange(node) && node < settings_.nodes - 1) {
                int df_idx = 0;
                for (const auto& frame : settings_.contact_frames) {
                    vectorx_t vio_vec = friction_cone_->GetViolation(traj_.GetForce(node, frame),
                        df.segment<CONTACT_3DOF>(df_idx));
                    df_idx += CONTACT_3DOF;

                    vio_vec *= in_contact_[frame][node];
                    for (int i = 0; i < 4; i++) {
                        vio_vec(i) = std::min(0., vio_vec(i));
                    }
                    // std::cout <<"[" << frame << "] Friction cone vio: |" << vio_vec.squaredNorm() << "| " << vio_vec.transpose() << std::endl;
                    violation += vio_vec.squaredNorm();
                }
            }

            if (swing_constraint_->IsInNodeRange(node)) {
                for (const auto& frame : settings_.contact_frames) {
                    vectorx_t vio_vec = swing_constraint_->GetViolation(traj_.GetConfiguration(node), dq,
                        swing_traj_[frame][node], frame);

                    // For the buffer
                    if (std::abs(vio_vec(0)) > settings_.swing_buffer) {
                        vio_vec(0) = std::abs(vio_vec(0)) - settings_.swing_buffer;
                    } else {
                        vio_vec(0) = 0;
                    }

                    // std::cout << "[" << frame << "] Swing vio: |" << vio_vec.squaredNorm() << "| " << vio_vec(0) << std::endl;
                    violation += vio_vec.squaredNorm();
                }
            }

            if (holonomic_->IsInNodeRange(node) && node < settings_.nodes - 1) {
                for (const auto& frame : settings_.contact_frames) {
                    vectorx_t vio_vec = holonomic_->GetViolation(traj_.GetConfiguration(node),
                        traj_.GetVelocity(node), dq, dv, frame);
                    vio_vec *= in_contact_[frame][node];
                    // std::cout << "[" << frame << "] holonomic vio: |" << vio_vec.squaredNorm() << "| " << vio_vec.transpose() << std::endl;
                    violation += vio_vec.squaredNorm();
                }
            }

            if (collision_->IsInNodeRange(node)) {
                for (int i = 0; i < collision_->GetNumCollisions(); i++) {
                    vectorx_t vio_vec = collision_->GetViolation(traj_.GetConfiguration(node), dq, i);
                    vio_vec(0) = std::min(0., vio_vec(0));
                    violation += vio_vec.squaredNorm();
                }
            }
        }


        // Compute average dt
        double dt_avg = 0;
        for (int i = 0; i < settings_.dt.size(); i++) {
            dt_avg += settings_.dt[i];
        }

        dt_avg /= settings_.dt.size();


        return dt_avg*std::sqrt(violation);
    }

    double HpipmMpc::GetCost(const std::vector<hpipm::OcpQpSolution> &sol, double alpha) {
        double cost = 0;
        for (int node = 0; node < settings_.nodes; node++) {
            vectorx_t dq = sol[node].x.head(nv_);
            vectorx_t dv(nv_);
            vectorx_t dtau = vectorx_t::Zero(ntau_);
            dv.head<FLOATING_VEL>() = sol[node].x.segment<FLOATING_VEL>(nv_);
            if (dynamics_constraint_->IsInNodeRange(node)) {
                dv.tail(ntau_) = sol[node].x.segment(FLOATING_VEL + nv_, ntau_);
                if (node < settings_.nodes - 1) {
                    dtau = sol[node].u.head(ntau_);
                }
            } else if (node  < settings_.nodes - 1) {
                dv.tail(ntau_) = sol[node].u.head(ntau_);
            }
            vectorx_t df = vectorx_t::Zero(CONTACT_3DOF*settings_.num_contact_locations);
            if (node < settings_.nodes - 1) {
                df = sol[node].u.tail(CONTACT_3DOF*settings_.num_contact_locations);
            }

            dq *= alpha;
            dv *= alpha;
            dtau *= alpha;
            df *= alpha;

            if (node < settings_.nodes - 1) {
                if (vel_tracking_->IsInNodeRange(node)) {
                    cost += settings_.dt[node]*vel_tracking_->GetCost(traj_.GetVelocity(node), dv, GetVelocityTarget(node),
                    dynamics_constraint_->IsInNodeRange(node) ?
                    settings_.cost_data[fo_vel_idx_].weight : settings_.cost_data[cent_vel_idx_].weight);
                }

                if (tau_tracking_->IsInNodeRange(node) && dynamics_constraint_->IsInNodeRange(node)) {
                    cost += settings_.dt[node]*tau_tracking_->GetCost(traj_.GetTau(node), dtau, GetTauTarget(node),
                        settings_.cost_data[fo_tau_idx_].weight);
                }

                if (force_tracking_->IsInNodeRange(node)) {
                    int f_idx = 0;
                    for (const auto& frame : settings_.contact_frames) {
                        cost += settings_.dt[node]*force_tracking_->GetCost(traj_.GetForce(node, frame),
                            df.segment<CONTACT_3DOF>(f_idx), GetForceTarget(node, frame),
                            dynamics_constraint_->IsInNodeRange(node) ?
                            settings_.cost_data[fo_force_idx_].weight : settings_.cost_data[cent_force_idx_].weight);
                        f_idx += CONTACT_3DOF;
                    }
                }

                if (config_tracking_->IsInNodeRange(node)) {
                    cost += settings_.dt[node]*config_tracking_->GetCost(traj_.GetConfiguration(node), dq,
                        GetConfigTarget(node),
                        dynamics_constraint_->IsInNodeRange(node) ?
                        settings_.cost_data[fo_config_idx_].weight : settings_.cost_data[cent_config_idx_].weight);
                }
            } else {
                cost += settings_.dt[node]*settings_.terminal_weight*vel_tracking_->GetCost(
                    traj_.GetVelocity(node), dv, GetVelocityTarget(node),
                    dynamics_constraint_->IsInNodeRange(node) ?
                        settings_.cost_data[fo_vel_idx_].weight : settings_.cost_data[cent_vel_idx_].weight);
                cost += settings_.dt[node]*settings_.terminal_weight*config_tracking_->GetCost(
                    traj_.GetConfiguration(node), dq, GetConfigTarget(node),
                    dynamics_constraint_->IsInNodeRange(node) ?
                        settings_.cost_data[fo_config_idx_].weight : settings_.cost_data[cent_config_idx_].weight);
            }
        }

        return std::sqrt(cost);
    }


    std::pair<double, double> HpipmMpc::LineSearch(const std::vector<hpipm::OcpQpSolution> &sol) {
        alpha_ = 1;

        double theta_k = GetConstraintViolation(sol, 0);
        double phi_k = GetCost(sol, 0);

        while (alpha_ > settings_.ls_alpha_min) {
            double theta_kp1 = GetConstraintViolation(sol, alpha_);

            if (theta_kp1 >= settings_.ls_theta_max) {
                if (theta_kp1 < (1 - settings_.ls_gamma_theta)*theta_k) {
                    // ls_condition_ = ConstraintViolation;
                    // std::cout << "CONSTRAINT REDUCTION" << std::endl;
                    // std::cout << "alpha = " << alpha_ << std::endl;
                    double phi_kp1 = GetCost(sol, alpha_);
                    return std::make_pair(theta_kp1, phi_kp1);
                }
            } else if (std::max(theta_k, theta_kp1) < settings_.ls_theta_min && SolutionGradientDot(sol) < 0) {
                double phi_kp1 = GetCost(sol, alpha_);
                if (phi_kp1 < (phi_k) + settings_.ls_eta*alpha_*SolutionGradientDot(sol)) {
                    // ls_condition_ = CostReduction;
                    // std::cout << "COST REDUCTION" << std::endl;
                    // std::cout << "alpha = " << alpha_ << std::endl;
                    return std::make_pair(theta_kp1, phi_kp1);
                }
            } else {
                double phi_kp1 = GetCost(sol, alpha_);
                if (phi_kp1 < (1 - settings_.ls_gamma_phi)*phi_k || theta_kp1 < (1 - settings_.ls_gamma_theta)*theta_k) {
                    // ls_condition_ = Both;
                    // std::cout << "CONSTRAINT & COST REDUCTION" << std::endl;
                    // std::cout << "alpha = " << alpha_ << std::endl;
                    return std::make_pair(theta_kp1, phi_kp1);
                }
            }
            alpha_ = settings_.ls_gamma_alpha*alpha_;
        }
        // ls_condition_ = MinAlpha;
        alpha_ = 0;

        return std::make_pair(theta_k, phi_k);
    }

    double HpipmMpc::SolutionGradientDot(const std::vector<hpipm::OcpQpSolution> &sol) {
        double prod = 0;
        for (int i = 0; i < settings_.nodes - 1; i++) {
            prod += qp[i].q.dot(sol[i].x) + qp[i].r.dot(sol[i].u);
        }

        prod += qp[settings_.nodes - 1].q.dot(sol[settings_.nodes - 1].x);

        return prod;
    }


    Trajectory HpipmMpc::GetTrajectory() const {
        return traj_;
    }

    int HpipmMpc::GetSolveCounter() const {
        return solve_counter_;
    }

    std::map<std::string, std::vector<double> > HpipmMpc::GetSwingTrajectory() const {
        return swing_traj_;
    }

    void HpipmMpc::LogMPCCompute(double time, const vectorx_t& q0, const vectorx_t& v0) {
        torc::utils::TORCTimer timer;
        timer.Tic();

        log_file_ << statistics_.iter << ",";
        if (result_ == hpipm::HpipmStatus::Success) {
            log_file_ << 1 << ",";
        } else if (result_ == hpipm::HpipmStatus::MaxIterReached) {
            log_file_ << 2 << ",";
        } else if (result_ == hpipm::HpipmStatus::MinStepLengthReached) {
            log_file_ << 3 << ",";
        } else {
            log_file_ << 4 << ",";
        }

        log_file_ << statistics_.obj[statistics_.obj.size()-2] << ",";
        log_file_ << statistics_.max_res_stat << "," << statistics_.max_res_eq << "," << statistics_.max_res_ineq << "," <<
            statistics_.max_res_comp << ",";
        log_file_ << solve_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << ",";
        log_file_ << alpha_ << "," << prev_violation_ << "," << prev_cost_ << ",";
        log_file_ << constraint_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << "," <<
            cost_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << "," <<
            ls_timer_.Duration<std::chrono::microseconds>().count()/1000.0 << ",";

        LogData(time, q0, v0);

        timer.Toc();
        if (settings_.verbose) {
            std::cout << "Logging took " << timer.Duration<std::chrono::microseconds>().count()/1000.0 << " ms" << std::endl;
        }
    }


    void HpipmMpc::LogData(double time, const vectorx_t &q, const vectorx_t &v) {
        // I can always put some of the logging back (at the expense of timing)

        // Solve number
        log_file_ << time << "," << solve_counter_ << "," << settings_.nodes << ",";

        // Initial condition
        LogEigenVec(q);
        LogEigenVec(v);

        // Computed traj & dt
        for (int i = 0; i < settings_.nodes; i++) {
            // LogEigenVec(traj_.GetConfiguration(i));
            // LogEigenVec(traj_.GetVelocity(i));
            // LogEigenVec(traj_.GetTau(i));
            // for (const auto& frame : settings_.contact_frames) {
            //     LogEigenVec(traj_.GetForce(i, frame));
            // }
            log_file_ << settings_.dt[i] <<  ",";
        }

        // Contact status
        for (int i = 0; i < settings_.nodes; i++) {
            for (const auto& frame : settings_.contact_frames) {
                log_file_ << in_contact_[frame][i] << ",";
            }
        }

        // // Swing traj
        // for (int i = 0; i < settings_.nodes; i++) {
        //     for (const auto& frame : settings_.contact_frames) {
        //         log_file_ << swing_traj_[frame][i] << ",";
        //     }
        // }

        // Current frame positions
        for (const auto& frame : settings_.contact_frames) {
            LogEigenVec(model_.GetFrameState(frame, q, v, pinocchio::WORLD).placement.translation().transpose());
        }

        // Current frame velocities
        for (const auto& frame : settings_.contact_frames) {
            LogEigenVec(model_.GetFrameState(frame, q, v, pinocchio::WORLD).vel.linear());
        }

        // // Target frame positions
        // for (int i = 0; i < settings_.nodes; i++) {
        //     for (const auto& frame : settings_.contact_frames) {
        //         LogEigenVec(end_effector_targets_[frame][i]);
        //     }
        // }

        // Optimized frame positions
        for (int i = 0; i < settings_.nodes; i++) {
            for (const auto& frame : settings_.contact_frames) {
                LogEigenVec(model_.GetFrameState(frame, traj_.GetConfiguration(i), traj_.GetVelocity(i),
                    pinocchio::WORLD).placement.translation().transpose());
            }
        }

        // Optimized frame velocities
        for (int i = 0; i < settings_.nodes; i++) {
            for (const auto& frame : settings_.contact_frames) {
                LogEigenVec(model_.GetFrameState(frame, traj_.GetConfiguration(i), traj_.GetVelocity(i),
                    pinocchio::WORLD).vel.linear());
            }
        }

        log_file_ << std::endl;
    }

    void HpipmMpc::LogEigenVec(const vectorx_t &x) {
        for (int i = 0; i < x.size(); i++) {
            log_file_ << x(i) << ",";
        }
    }



    void HpipmMpc::PrintNodeInfo() const {
        std::cout << std::setfill(' ');
        std::cout << std::setw(22) << "";
        for (int node = 0; node < settings_.nodes; node++) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(2) << node;
        }
        std::cout << std::endl;
        std::cout << std::setw(22) << "";
        double time = 0.00;
        for (int node = 0; node < settings_.nodes; node++) {
            std::cout << std::setw(1) << std::fixed << std::setprecision(2) << " " << time;

            time += settings_.dt[node];
        }

        for (const auto& frame : settings_.contact_frames) {
            std::cout << std::endl;

            std::cout << std::setw(20) << frame << "  ";
            for (int node = 0; node < settings_.nodes; node++) {
                if (in_contact_.at(frame)[node]) {
                    std::cout << " ____";
                } else {
                    std::cout << " xxxx";
                }
            }
            std::cout << std::endl;

            // Reset precision and formatting to default
            std::cout.unsetf(std::ios::fixed);
            std::cout.precision(6); // Default precision (usually 6)
        }

        std::cout << std::setw(20) << "Dynamics" << "  ";
        for (int node = 0; node < settings_.nodes; node++) {
            if (dynamics_constraint_->IsInNodeRange(node)) {
                std::cout << std::setw(1) << std::fixed << std::setprecision(2) << " " << "FO  " ;
            } else {
                std::cout << std::setw(1) << std::fixed << std::setprecision(2) << " " << "ROM ";
            }
        }
        std::cout << std::endl;

        std::cout << std::setw(20) << "Friction" << "  ";
        for (int node = 0; node < settings_.nodes; node++) {
            if (friction_cone_->IsInNodeRange(node)) {
                std::cout << std::setw(1) << std::fixed << std::setprecision(2) << " " << "T   " ;
            } else {
                std::cout << std::setw(1) << std::fixed << std::setprecision(2) << " " << "F   ";
            }
        }
        std::cout << std::endl;

        std::cout << std::setw(20) << "Swing" << "  ";
        for (int node = 0; node < settings_.nodes; node++) {
            if (swing_constraint_->IsInNodeRange(node)) {
                std::cout << std::setw(1) << std::fixed << std::setprecision(2) << " " << "T   " ;
            } else {
                std::cout << std::setw(1) << std::fixed << std::setprecision(2) << " " << "F   ";
            }
        }
        std::cout << std::endl;

        std::cout << std::setw(20) << "Holonomic" << "  ";
        for (int node = 0; node < settings_.nodes; node++) {
            if (holonomic_->IsInNodeRange(node)) {
                std::cout << std::setw(1) << std::fixed << std::setprecision(2) << " " << "T   " ;
            } else {
                std::cout << std::setw(1) << std::fixed << std::setprecision(2) << " " << "F   ";
            }
        }
        std::cout << std::endl;

        for (const auto& frame : settings_.contact_frames) {
            std::cout << std::setw(20) << " ";
            for (int node = 0; node < settings_.nodes; node++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << node;
            }
            std::cout << std::endl;
            std::cout << std::setw(20) << "Force [" + frame + "] x";
            for (int node = 0; node < settings_.nodes; node++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << traj_.GetForce(node, frame)[0];
            }
            std::cout << std::endl;
            std::cout << std::setw(20) << "Force [" + frame + "] y";
            for (int node = 0; node < settings_.nodes; node++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << traj_.GetForce(node, frame)[1];
            }
            std::cout << std::endl;
            std::cout << std::setw(20) << "Force [" + frame + "] z";
            for (int node = 0; node < settings_.nodes; node++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << traj_.GetForce(node, frame)[2];
            }
            std::cout << std::endl;
        }
    }
}

