//
// Created by zolkin on 2/3/25.
//

#include "yaml-cpp/yaml.h"

#include "pinocchio_interface.h"
#include "wbc_controller.h"

#include <eigen_utils.h>
#include <torc_timer.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>

namespace torc::controller {
    WbcSettings::WbcSettings(const std::filesystem::path& config_file) {
        // Read in the yaml file.
        YAML::Node config;
        try {
            config = YAML::LoadFile(config_file);
        } catch (...) {
            throw std::runtime_error("Could not load the configuration file!");
        }

        // ---------- Joint Default Settings ---------- //
        if (config["wbc"]) {
            std::cout << "Loading WBC Settings..." << std::endl;
            YAML::Node wbc_settings = config["wbc"];

            std::vector<double> base_weight_vec = wbc_settings["base_weight"].as<std::vector<double>>();
            base_weight = utils::StdToEigenVector(base_weight_vec);

            std::vector<double> joint_weight_vec = wbc_settings["joint_weight"].as<std::vector<double>>();
            joint_weight = utils::StdToEigenVector(joint_weight_vec);

            std::vector<double> tau_weight_vec = wbc_settings["tau_weight"].as<std::vector<double>>();
            tau_weight = utils::StdToEigenVector(tau_weight_vec);

            std::vector<double> force_weight_vec = wbc_settings["force_weight"].as<std::vector<double>>();
            force_weight = utils::StdToEigenVector(force_weight_vec);

            std::vector<double> kp_vec = wbc_settings["kp"].as<std::vector<double>>();
            kp = utils::StdToEigenVector(kp_vec);

            std::vector<double> kd_vec = wbc_settings["kd"].as<std::vector<double>>();
            kd = utils::StdToEigenVector(kd_vec);

            if (wbc_settings["custom_torque_lims"]) {
                std::vector<double> torque_lims = wbc_settings["custom_torque_lims"].as<std::vector<double>>();
                custom_torque_lims = utils::StdToEigenVector(torque_lims);
            }

            verbose = wbc_settings["verbose"].as<bool>();

            // Load in the frame tracking settings
            if (wbc_settings["frame_tracking"]) {
                for (const auto& frame_term : wbc_settings["frame_tracking"]) {
                    tracking_frames_.emplace_back(frame_term["frame"].as<std::string>());
                    tracking_weights_.emplace_back(utils::StdToEigenVector(frame_term["weight"].as<std::vector<double>>()));
                    tracking_kp.emplace_back(utils::StdToEigenVector(frame_term["kp"].as<std::vector<double>>()));
                    tracking_kd.emplace_back(utils::StdToEigenVector(frame_term["kd"].as<std::vector<double>>()));
                }
            }

            skip_joints = wbc_settings["skip_joints"].as<std::vector<std::string>>();
            joint_values = wbc_settings["joint_values"].as<std::vector<double>>();
            contact_frames = wbc_settings["contact_frames"].as<std::vector<std::string>>();
            compile_derivs = wbc_settings["compile_derivs"].as<bool>();
            alpha = wbc_settings["alpha"].as<double>();
            log = wbc_settings["log"].as<bool>();
            log_period = wbc_settings["log_period"].as<int>();
        } else {
            throw std::runtime_error("[WBC] No settings found in yaml file!");
        }

    }

    WbcController::WbcController(const models::FullOrderRigidBody &model,
        const std::vector<std::string> &contact_frames, WbcSettings settings, double friction_coef, bool verbose,
        const std::filesystem::path deriv_lib_path, bool compile_derivs)
            : model_(model), contact_frames_(contact_frames), verbose_(verbose), pin_data_(model.GetModel()),
                mu_(friction_coef), settings_(std::move(settings)), solve_count_(0) {

        if (settings_.log) {
            log_file_.open("wbc_log_file.csv"); // TODO: Allow the name and location to be modified
        }

        nv_ = model_.GetVelDim();
        nq_ = model_.GetConfigDim();
        ntau_ = nv_ - FLOATING_VEL;
        ncontact_frames_ = contact_frames_.size();
        nF_ = CONTACT_3DOF*ncontact_frames_;
        nd_ = nv_ + ntau_ + nF_;

        if (settings_.kd.size() != settings_.kp.size() || settings_.kp.size() != nv_) {
            std::cerr << "kd size: " << settings_.kd.size() << std::endl;
            std::cerr << "kp size: " << settings_.kp.size() << std::endl;
            std::cerr << "nv size: " << nv_ << std::endl;
            throw std::runtime_error("[WBC] kd and kp sizes do not match or they do not match the model!");
        }

        if (settings_.base_weight.size() != 6) {
            throw std::runtime_error("[WBC] base_weight size must be 6!");
        }

        if (settings_.tau_weight.size() != ntau_) {
            std::cerr << "tau weight size: " << settings_.tau_weight.size() << std::endl;
            std::cerr << "ntau: " << ntau_ << std::endl;
            throw std::runtime_error("[WBC] tau_weight size does not match!");
        }

        // Make the auto diff function
        inverse_dynamics_ = std::make_unique<ad::CppADInterface>(
            std::bind(&WbcController::DynamicsFunction, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3),
            "wbc_dynamics_constraint",
            deriv_lib_path,
            ad::DerivativeOrder::FirstOrder, nd_, model_.GetConfigDim() + nv_,
            compile_derivs
        );

        if (settings_.custom_torque_lims.size() != 0 && settings_.custom_torque_lims.size() != ntau_) {
            throw std::runtime_error("Custom torque limits are not the correct size!");
        }
    }

    vectorx_t WbcController::ComputeControl(const vectorx_t &q, const vectorx_t &v,
        const vectorx_t &q_des, const vectorx_t &v_des, const vectorx_t &tau_des, const vectorx_t &F_des,
        std::vector<bool> in_contact) {
        torc::utils::TORCTimer timer;

        if (settings_.log && solve_count_ % settings_.log_period == 0) {
            log_file_ << solve_count_ << ",";
        }

        // std::cout << "q: " << q.transpose() << std::endl;
        // std::cout << "q des: " << q_des.transpose() << std::endl;

        timer.Tic();
        if (in_contact.size() != ncontact_frames_) {
            throw std::runtime_error("[WBC] contact vector does not match contact frames size!");
        }

        if (q.size() != nq_) {
            throw std::runtime_error("[WBC] incorrect q size! Expected " + std::to_string(nq_) + " got: " + std::to_string(q.size()));
        }
        if (v.size() != nv_) {
            throw std::runtime_error("[WBC] incorrect v size! Expected " + std::to_string(nv_) + " got: " + std::to_string(v.size()));
        }
        if (tau_des.size() != ntau_) {
            throw std::runtime_error("[WBC] incorrect tau_des size! Expected " + std::to_string(ntau_) + " got: " + std::to_string(tau_des.size()));
        }
        if (F_des.size() != nF_) {
            throw std::runtime_error("[WBC] incorrect F_des size! Expected " + std::to_string(nF_) + " got: " + std::to_string(F_des.size()));
        }
        if (q_des.size() != nq_) {
            throw std::runtime_error("[WBC] incorrect q_des size! Expected " + std::to_string(nq_) + " got: " + std::to_string(q_des.size()));
        }
        if (v_des.size() != nv_) {
            throw std::runtime_error("[WBC] incorrect v_des size! Expected " + std::to_string(nv_) + " got: " + std::to_string(v_des.size()));
        }

        // For now, updating the size with every solve depending on the contacts
        ncontacts_ = 0;
        for (const auto& contact : in_contact) {
            if (contact) {
                ncontacts_++;
            }
        }
        // std::cerr << "tau: " << tau_des.transpose() << std::endl;

        // Dynamics, no-slip, friction cone, no force, torque box
        nc_ = nv_ + ncontacts_*3 + ncontacts_*5 + (ncontact_frames_ - ncontacts_)*3 + ntau_;
        int neq = nv_ + ncontacts_*3 + (ncontact_frames_ - ncontacts_)*3;
        int nineq = ncontacts_*5 + ntau_;

        // ProxQpInterface
        proxsuite::proxqp::dense::QP<double> qp(nd_, neq, nineq);
        matrixx_t H = matrixx_t::Zero(nd_, nd_);
        vectorx_t g = vectorx_t::Zero(nd_);
        matrixx_t Aeq = matrixx_t::Zero(neq, nd_);
        vectorx_t beq = vectorx_t::Zero(neq);
        matrixx_t Cineq = matrixx_t::Zero(nineq, nd_);
        vectorx_t uineq = vectorx_t::Zero(nineq);
        vectorx_t lineq = vectorx_t::Zero(nineq);

        // --------- Constraints ---------- //

        // Equality
        int rowc = 0;
        // Dynamics
        vectorx_t b(nv_);
        Aeq.topRows(nv_) = DynamicsConstraint(q, v, b);
        beq.topRows(nv_) = b;
        rowc += nv_;

        // No Force in swing
        b.resize((ncontact_frames_ - ncontacts_)*3);
        Aeq.middleRows(rowc, (ncontact_frames_ - ncontacts_)*3) = NoForceConstraint(in_contact, b);
        beq.segment(rowc, (ncontact_frames_ - ncontacts_)*3) = b;
        rowc += (ncontact_frames_ - ncontacts_)*3;

        // Holonomic
        b.resize(ncontacts_*3);
        Aeq.middleRows(rowc, ncontacts_*3) = HolonomicConstraint(q, v, in_contact, b);
        beq.segment(rowc, ncontacts_*3) = b;
        rowc += 3*ncontacts_;


        // // Inequality
        rowc = 0;
        // Friction Cone
        vectorx_t ub, lb;
        ub.resize(5*ncontacts_);
        lb.resize(5*ncontacts_);
        Cineq.middleRows(rowc, 5*ncontacts_) = ForceConstraint(in_contact, lb, ub);
        uineq.segment(0, 5*ncontacts_) = ub;
        lineq.segment(0, 5*ncontacts_) = lb;
        rowc += 5*ncontacts_;

        // Torque box
        Cineq.middleRows(rowc, ntau_) = TorqueBoxConstraint();
        if (settings_.custom_torque_lims.size() == 0) {
            uineq.segment(rowc, ntau_) = model_.GetTorqueJointLimits().tail(ntau_);
            lineq.segment(rowc, ntau_) = -model_.GetTorqueJointLimits().tail(ntau_);
        } else {
            uineq.segment(rowc, ntau_) = settings_.custom_torque_lims;
            lineq.segment(rowc, ntau_) = -settings_.custom_torque_lims;
        }
        // --------- Cost ---------- //
        const auto [stateP, stateq] = StateTracking(q, v, q_des, v_des);
        H.topLeftCorner(nv_, nv_) = stateP;
        g.head(nv_) = stateq;

        const auto [trackingP, trackingq] = FrameTracking(q, v, q_des, v_des,
            in_contact);
        H.topLeftCorner(nv_, nv_) += trackingP;
        g.head(nv_) += trackingq;

        const auto [tauP, tauq] = TorqueTracking(tau_des);
        H.block(nv_, nv_, ntau_, ntau_) = tauP;
        g.segment(nv_, ntau_) = tauq;

        const auto [FP, Fq] = ForceTracking(F_des);
        H.bottomRightCorner(CONTACT_3DOF*ncontact_frames_, CONTACT_3DOF*ncontact_frames_) = FP;
        g.segment(nv_ + ntau_, nF_) = Fq;

        // --------- Solve ---------- //
        qp.init(H, g, Aeq, beq, Cineq, lineq, uineq);
        timer.Toc();

        utils::TORCTimer solve_timer;
        solve_timer.Tic();
        qp.solve();
        solve_timer.Toc();

        if (verbose_) {
            // std::cout << "num contacts: " << ncontacts_ << std::endl;
            // std::cout << "tau: " << qp.results.x.segment(nv_, ntau_).transpose() << std::endl;
            // std::cout << "a: " << qp.results.x.head(nv_).transpose() << std::endl;
            // std::cout << "F: " << qp.results.x.segment(nv_ + ntau_, nF_).transpose() << std::endl;
            std::cout << "Setup took " << timer.Duration<std::chrono::microseconds>().count()*1e-3 << " ms" << std::endl;
            std::cout << "Solve [T1] took " << solve_timer.Duration<std::chrono::microseconds>().count()*1e-3 << " ms" << std::endl;
            std::cout << "Solve [T2] took " << qp.results.info.solve_time << " ms" << std::endl;
        }

        if (qp.results.info.status != proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
            std::cerr << "iters: " << qp.results.info.iter << std::endl;
            std::cerr << "run time: " << qp.results.info.run_time << std::endl;
            std::cerr << "status: ";
            if (qp.results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_NOT_RUN) {
                std::cerr << "NOT RUN" << std::endl;
            } else if (qp.results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_DUAL_INFEASIBLE) {
                std::cerr << "Dual infeasible" << std::endl;
            } else if (qp.results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_MAX_ITER_REACHED) {
                std::cerr << "MAX_ITER_REACHED" << std::endl;
            } else if (qp.results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_PRIMAL_INFEASIBLE) {
                std::cerr << "PRIMAL_INFEASIBLE" << std::endl;
            } else if (qp.results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE) {
                std::cerr << "SOLVED_CLOSEST_PRIMAL_FEASIBLE" << std::endl;
            }
            std::cerr << "primal resid: " << qp.results.info.pri_res << std::endl;
            std::cerr << "dual resid: " << qp.results.info.dua_res << std::endl;
            throw std::runtime_error("WBC not solved!");
        }

        if (settings_.log && solve_count_ % settings_.log_period == 0) {
            LogEigenVec(qp.results.x);
            for (const auto& contact : in_contact) {
                log_file_ << contact << ",";
            }
            log_file_ << std::endl;
        }

        // Check constraint violation
        // const vectorx_t violation = A*osqp_solver.primal_solution();
        // for (int i = 0; i < violation.size(); i++) {
        //     if (violation(i) > osqp_instance_.upper_bounds(i)) {
        //         std::cout << violation(i) - osqp_instance_.upper_bounds(i) << " ";
        //     } else if (violation(i) < osqp_instance_.lower_bounds(i)) {
        //         std::cout << osqp_instance_.lower_bounds(i) - violation(i) << " ";
        //     } else {
        //         std::cout << "0 ";
        //     }
        // }
        // std::cout << std::endl;

        // Now check the actual functions to verify there was no transcription error
        vectorx_t p(nq_ + nv_);
        p << q, v;

        vectorx_t y;
        inverse_dynamics_->GetFunctionValue(qp.results.x, p, y);
        // if (y.norm() > 1e-4) {
        //     std::cout << "dynamics vio: " << y.transpose() << std::endl;
        //     pinocchio::computeAllTerms(model_.GetModel(), pin_data_, q, v);
        //     pin_data_.M.triangularView<Eigen::StrictlyLower>() = pin_data_.M.transpose().triangularView<Eigen::StrictlyLower>();
        //
        //     // Check Force matrix
        //     // Compute the frame jacobians
        //     matrixx_t J = matrixx_t::Zero(3*ncontact_frames_, nv_);
        //     for (int i = 0 ; i < ncontact_frames_; i++) {
        //         matrixx_t Jtemp = matrixx_t::Zero(6, nv_);
        //         pinocchio::computeFrameJacobian(model_.GetModel(), pin_data_, q,
        //             model_.GetFrameIdx(contact_frames_[i]), pinocchio::LOCAL, Jtemp); // TODO: Play with the frames
        //         J.middleRows<3>(3*i) = Jtemp.topRows<3>();
        //     }
        //
        //     matrixx_t tau_mat = matrixx_t::Zero(nv_, ntau_);
        //     tau_mat.bottomRows(ntau_).setIdentity();
        //     tau_mat *= -1;
        //
        //     vectorx_t vio = pin_data_.M*osqp_solver.primal_solution().head(nv_) +
        //         tau_mat*osqp_solver.primal_solution().segment(nv_, ntau_) +
        //             -J.transpose()*osqp_solver.primal_solution().tail(nF_) + pin_data_.nle;
        //     std::cout << "analytic vio: " << vio.transpose() << std::endl;
        //
        //     // throw std::runtime_error("Dynamics violation of " + std::to_string(y.norm()));
        // }

        // pinocchio::forwardKinematics(model_.GetModel(), pin_data_, q, v, osqp_solver.primal_solution().head(nv_));
        // for (const auto& frame : contact_frames_) {
        //     int frame_idx = model_.GetFrameIdx(frame);
        //     vectorx_t fal = pinocchio::getFrameAcceleration(model_.GetModel(), pin_data_, frame_idx).linear();
        //     vectorx_t faw = pinocchio::getFrameAcceleration(model_.GetModel(), pin_data_, frame_idx).angular();
        //     vectorx_t fv = pinocchio::getFrameVelocity(model_.GetModel(), pin_data_, frame_idx).linear();
        //     // if (fv.norm() > 1e-2) {
        //     //     std::cout << frame << " has vel " << fv.transpose() << std::endl;
        //     // }
        //     if (fal.norm() > 1e-6) {
        //         std::cout << frame << " has linear accel " << fal.transpose() << std::endl;
        //         // throw std::runtime_error(frame + " has accel " + std::to_string(fa.norm()));
        //     }
        //     if (faw.norm() > 1e-6) {
        //         std::cout << frame << " has angular accel " << faw.transpose() << std::endl;
        //         // throw std::runtime_error(frame + " has accel " + std::to_string(fa.norm()));
        //     }
        //     // std::cout << frame << " accel: " << fa.transpose() << std::endl;
        // }

        // std::cout << "vio: " << violation.transpose() << std::endl;

        solve_count_++;

        // Return the torques
        return qp.results.x.segment(nv_, ntau_);
    }

    matrixx_t WbcController::HolonomicConstraint(const vectorx_t &q, const vectorx_t &v,
        const std::vector<bool> &in_contact, vectorx_t& b) {
        matrixx_t A = matrixx_t::Zero(3*ncontacts_, nd_);

        pinocchio::forwardKinematics(model_.GetModel(), pin_data_, q, v, vectorx_t::Zero(nv_));

        int A_row = 0;
        for (int i = 0; i < in_contact.size(); i++) {
            if (in_contact[i]) {
                // TODO: Put back to all 3

                // Get the frame Jacobian
                matrix6_t J = matrix6_t::Zero(6, nv_);
                model_.GetFrameJacobian(contact_frames_.at(i), q, J, pinocchio::LOCAL);
                A.block(A_row, 0, 3, nv_) = J.topRows<3>();

                // // Get the Jdot term
                b.segment<3>(A_row) = -pinocchio::getFrameAcceleration(model_.GetModel(), pin_data_,
                    model_.GetFrameIdx(contact_frames_.at(i)), pinocchio::LOCAL).linear().head<3>();
                // b.segment<3>(A_row + 3) = -pinocchio::getFrameAcceleration(model_.GetModel(), pin_data_,
                //     model_.GetFrameIdx(contact_frames_.at(i)), pinocchio::LOCAL).angular().head<3>();
                // b.segment<6>(A_row) -= settings_.alpha*J*v;
                // lb(A_row + 2) = 0;
                A_row += 3;
            }
        }

        return A;
    }

    matrixx_t WbcController::ForceConstraint(const std::vector<bool> &in_contact, vectorx_t &lb, vectorx_t &ub) {
        matrixx_t A = matrixx_t::Zero(5*ncontacts_, nd_);

        int row_idx = 0;
        for (int i = 0; i < in_contact.size(); i++) {
            if (in_contact[i]) {
                // Linear friction cone
                A.block(row_idx, nv_ + ntau_ + 3*i, 5, 3) <<
                    -1, 0, mu_,
                    1, 0, mu_,
                    0, -1, mu_,
                    0, 1, mu_,
                    0, 0, 1;

                lb.segment<4>(row_idx).setZero();
                ub.segment<4>(row_idx) = vector4_t::Constant(4, 1000);
                row_idx += 4;

                lb(row_idx) = 10;
                ub(row_idx) = 1000;
                row_idx++;
            }
        }

        return A;
    }

    matrixx_t WbcController::NoForceConstraint(const std::vector<bool> &in_contact, vectorx_t &b) {
        matrixx_t A = matrixx_t::Zero(3*(ncontact_frames_ - ncontacts_), nd_);

        int row_idx = 0;
        for (int i = 0; i < in_contact.size(); i++) {
            if (!in_contact[i]) {
                A.block(row_idx, nv_ + ntau_ + 3*i, 3, 3).setIdentity();

                b.segment<3>(row_idx).setZero();

                row_idx += 3;
            }
        }

        return A;
    }


    matrixx_t WbcController::DynamicsConstraint(const vectorx_t &q, const vectorx_t &v, vectorx_t& b) {
        matrixx_t A(nv_, nd_);
        // Compute the terms via auto diff
        vectorx_t x_zero = vectorx_t::Zero(nd_);    // Linearize about 0, but it doesn't matter bc its already linear

        vectorx_t p(model_.GetConfigDim() + nv_);
        p << q , v;

        inverse_dynamics_->GetJacobian(x_zero, p, A);

        // Get the bounds
        vectorx_t y;
        inverse_dynamics_->GetFunctionValue(x_zero, p, y);

        // TODO: Consider going to the analytic fully
        pinocchio::computeAllTerms(model_.GetModel(), pin_data_, q, v);
        pin_data_.M.triangularView<Eigen::StrictlyLower>() = pin_data_.M.transpose().triangularView<Eigen::StrictlyLower>();

        A.leftCols(nv_) = pin_data_.M;  // TODO: Remove after debugging

        // // Check M == A
        // std::cout << "M:\n" << pin_data_.M << std::endl;
        // std::cout << "A:\n" << A.leftCols(nv_) << std::endl;
        //
        // // Check nonlinear terms:
        // std::cout << "nle: " << pin_data_.nle.transpose() << std::endl;
        // std::cout << "y: " << y.transpose() << std::endl;
        y = pin_data_.nle;  // TODO: Remove after debugging
        //
        // // Check tau matrix
        // std::cout << "Atau:\n" << A.middleCols(nv_, ntau_) << std::endl;

        // Check Force matrix
        // Compute the frame jacobians
        matrixx_t J = matrixx_t::Zero(3*ncontact_frames_, nv_);
        for (int i = 0 ; i < ncontact_frames_; i++) {
            matrixx_t Jtemp = matrixx_t::Zero(6, nv_);
            pinocchio::computeFrameJacobian(model_.GetModel(), pin_data_, q,
                model_.GetFrameIdx(contact_frames_[i]), pinocchio::LOCAL, Jtemp); // TODO: Play with the frames
            J.middleRows<3>(3*i) = Jtemp.topRows<3>();
        }
        // std::cout << "Jforce:\n" << -J.transpose() << std::endl;
        // std::cout << "Aforce:\n" << A.rightCols(nF_) << std::endl;
        A.rightCols(nF_) = -J.transpose();  // TODO: Remove after debugging

        b = -y;
        return A;
    }

    void WbcController::DynamicsFunction(const ad::ad_vector_t &a_tau_F, const ad::ad_vector_t &q_v, ad::ad_vector_t &violation) {

        const ad::ad_vector_t& a = a_tau_F.head(nv_);
        const ad::ad_vector_t& tau = a_tau_F.segment(nv_, ntau_);
        const ad::ad_vector_t& F = a_tau_F.tail(nF_);

        const ad::ad_vector_t& q = q_v.head(model_.GetConfigDim());
        const ad::ad_vector_t& v = q_v.tail(nv_);

        // Make the force object
        std::vector<models::ExternalForce<ad::adcg_t>> f_ext;
        int idx = 0;
        for (const auto& frame : contact_frames_) {
            f_ext.emplace_back(frame, F.segment<CONTACT_3DOF>(idx));
            idx += CONTACT_3DOF;
        }

        // Call the inverse dynamics
        const ad::ad_vector_t tau_id = models::InverseDynamics(model_.GetADPinModel(), *model_.GetADPinData(),
            q, v, a, f_ext);

        violation = tau_id;
        violation.tail(ntau_) -= tau;
    }

    matrixx_t WbcController::TorqueBoxConstraint() const {
        matrixx_t A = matrixx_t::Zero(ntau_, nd_);
        A.block(0, nv_, ntau_, ntau_).setIdentity();

        return A;
    }

    std::pair<matrixx_t, vectorx_t> WbcController::FrameTracking(const vectorx_t &q, const vectorx_t &v,
        const vectorx_t &q_des, const vectorx_t &v_des, const std::vector<bool>& in_contact) {
        matrixx_t P = matrixx_t::Zero(nv_, nv_);
        vectorx_t g = vectorx_t::Zero(nv_);

        // TODO: Move all the .at(i) to [i]
        for (int i = 0; i < settings_.tracking_frames_.size(); i++) {
            // if (!in_contact[i]) { // NOTE: This assumes the frames are in the same order as the contacts!!
            const int frame_idx = model_.GetFrameIdx(settings_.tracking_frames_.at(i));

            // Compute the desired position and velocity
            pinocchio::forwardKinematics(model_.GetModel(), pin_data_, q_des, v_des);
            pinocchio::updateFramePlacement(model_.GetModel(), pin_data_, frame_idx);
            const vector3_t p_des = pin_data_.oMf[frame_idx].translation();
            const vector3_t pv_des = pinocchio::getFrameVelocity(model_.GetModel(), pin_data_, frame_idx, pinocchio::WORLD).linear();

            // Compute current position and velocity
            pinocchio::forwardKinematics(model_.GetModel(), pin_data_, q, v, vectorx_t::Zero(nv_));
            pinocchio::updateFramePlacement(model_.GetModel(), pin_data_, frame_idx);
            const vector3_t p_curr = pin_data_.oMf[frame_idx].translation();
            const vector3_t pv_curr = pinocchio::getFrameVelocity(model_.GetModel(), pin_data_, frame_idx, pinocchio::WORLD).linear();
            const vector3_t frame_acc_terms = pinocchio::getFrameAcceleration(model_.GetModel(), pin_data_, frame_idx, pinocchio::WORLD).linear();

            // Compute desired control
            vector3_t a_control = settings_.tracking_kp.at(i).asDiagonal()*(p_des - p_curr) +
                settings_.tracking_kd[i].asDiagonal()*(pv_des - pv_curr);

            if (settings_.log && solve_count_ % settings_.log_period == 0) {
                LogEigenVec(p_curr);
                LogEigenVec(pv_curr);
                LogEigenVec(p_des);
                LogEigenVec(pv_des);
                LogEigenVec(a_control);
            }
            // std::cout << "frame: " << settings_.tracking_frames_.at(i) << ", a control: " << a_control.transpose() << std::endl;
            // std::cout << "pos error: " << (p_des - p_curr).transpose() << ", vel error: " << (pv_des - pv_curr).transpose() << std::endl;
            // std::cout << "p des: " << p_des.transpose() << ", p curr: " << p_curr.transpose() << std::endl;

            // Compute Jacobian at the current config
            matrix6_t J;
            model_.GetFrameJacobian(settings_.tracking_frames_.at(i), q, J, pinocchio::WORLD);

            // TODO: Move the if state to capture everything
            if (!in_contact[i]) {
                // Compute linear term
                g += 2*J.topRows<3>().transpose()*settings_.tracking_weights_.at(i).asDiagonal()*(frame_acc_terms - a_control);

                // Quadratic term
                P += 2*J.topRows<3>().transpose()*settings_.tracking_weights_.at(i).asDiagonal()*J.topRows<3>();
            }
        }

        return {P, g};
    }

    std::pair<matrixx_t, vectorx_t> WbcController::StateTracking(const vectorx_t &q, const vectorx_t &v,
        const vectorx_t &q_des, const vectorx_t &v_des) {
        // Compute desired acceleration
        const vectorx_t q_diff = models::qDifference(q_des, q);
        const vectorx_t v_diff = v_des - v;

        const vectorx_t a_des = settings_.kp.asDiagonal()*q_diff + settings_.kd.asDiagonal()*v_diff;

        // Compute the mats
        matrixx_t P = matrixx_t::Zero(nv_, nv_);
        P.topLeftCorner<FLOATING_VEL, FLOATING_VEL>() = 2*settings_.base_weight.asDiagonal();
        P.bottomRightCorner(nv_ - FLOATING_VEL, nv_ - FLOATING_VEL) = 2*settings_.joint_weight.asDiagonal();

        vectorx_t g(nv_);
        g.head<FLOATING_VEL>().noalias() = settings_.base_weight.asDiagonal()*-2*a_des.head<FLOATING_VEL>();
        g.tail(nv_ - FLOATING_VEL).noalias() = settings_.joint_weight.asDiagonal()*-2*a_des.tail(nv_ - FLOATING_VEL);

        if (settings_.log && solve_count_ % settings_.log_period == 0) {
            LogEigenVec(q);
            LogEigenVec(v);
            LogEigenVec(q_diff);
            LogEigenVec(v_diff);
            LogEigenVec(a_des);
        }

        return {P, g};
    }

    std::pair<matrixx_t, vectorx_t> WbcController::TorqueTracking(const vectorx_t &tau_des) {
        matrixx_t P = 2*settings_.tau_weight.asDiagonal();
        vectorx_t g = settings_.tau_weight.asDiagonal()*-2*tau_des;

        if (settings_.log && solve_count_ % settings_.log_period == 0) {
            LogEigenVec(tau_des);
        }

        return {P, g};
    }

    std::pair<matrixx_t, vectorx_t> WbcController::ForceTracking(const vectorx_t &F_des) {
        matrixx_t P = 2*settings_.force_weight.asDiagonal();
        vectorx_t g = settings_.force_weight.asDiagonal()*-2*F_des;

        if (settings_.log && solve_count_ % settings_.log_period == 0) {
            LogEigenVec(F_des);
        }

        return {P, g};
    }

    void WbcController::LogEigenVec(const vectorx_t &vec) {
        for (int i = 0; i < vec.size(); i++) {
            log_file_ << vec(i) << ",";
        }
    }


    WbcController::~WbcController() {
        if (settings_.log) {
            log_file_.close();
        }
    }


    // Auto diff Frame Tracking:
    // void WbcController::FrameTrackingFunction(const std::string &frame, const ad::ad_vector_t &a,
    //     const ad::ad_vector_t &q_v_kp_kd_qd_vd_weights, ad::ad_vector_t &cost) {
    //
    //     const ad::ad_vector_t& q = q_v_kp_kd_qd_vd_weights.head(nq_);
    //     const ad::ad_vector_t& v = q_v_kp_kd_qd_vd_weights.segment(nq_, nv_);
    //     const ad::ad_vector3_t& kp = q_v_kp_kd_qd_vd_weights.segment<3>(nq_ + nv_);
    //     const ad::ad_vector3_t& kd = q_v_kp_kd_qd_vd_weights.segment<3>(nq_ + nv_ + 3);
    //     const ad::ad_vector_t& q_des = q_v_kp_kd_qd_vd_weights.segment(nq_ + nv_ + 6, nq_);
    //     const ad::ad_vector_t& v_des = q_v_kp_kd_qd_vd_weights.segment(nq_ + nv_ + 6 + nq_, nv_);
    //     const ad::ad_vector_t& weights = q_v_kp_kd_qd_vd_weights.tail<3>();
    //
    //     pinocchio::forwardKinematics(model_.GetADPinModel(), *model_.GetADPinData(),
    //         q, v, a);
    //
    //     int frame_idx = model_.GetFrameIdx(frame);
    //
    //     const ad::ad_vector3_t a_curr = pinocchio::getFrameAcceleration(model_.GetADPinModel(), *model_.GetADPinData(),
    //         frame_idx, pinocchio::WORLD).linear();
    //
    //     // Compute the desired position and velocity
    //     pinocchio::forwardKinematics(model_.GetADPinModel(), *model_.GetADPinData(), q_des, v_des);
    //     pinocchio::updateFramePlacement(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx);
    //     const ad::ad_vector3_t p_des = model_.GetADPinData()->oMf[frame_idx].translation();
    //     const ad::ad_vector3_t pv_des = pinocchio::getFrameVelocity(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx,
    //         pinocchio::WORLD).linear();
    //
    //     // Compute current position and velocity
    //     pinocchio::forwardKinematics(model_.GetADPinModel(), *model_.GetADPinData(), q, v, ad::ad_vector_t::Zero(nv_));
    //     pinocchio::updateFramePlacement(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx);
    //     const ad::ad_vector3_t p_curr = model_.GetADPinData()->oMf[frame_idx].translation();
    //     const ad::ad_vector3_t pv_curr = pinocchio::getFrameVelocity(model_.GetADPinModel(), *model_.GetADPinData(),
    //         frame_idx, pinocchio::WORLD).linear();
    //
    //
    //     const ad::ad_vector3_t a_des = kp.asDiagonal()*(p_des - p_curr) + kd.asDiagonal()*(pv_des - pv_curr);
    //
    //     cost = a_curr - a_des;
    //
    //     for (int i = 0; i < 3; i++) {
    //         cost[i] = cost[i]*weights[i];
    //     }
    // }

    // vectorx_t x_zero = vectorx_t::Zero(frame_tracking_[settings_.tracking_frames_[i]]->GetDomainSize());
    // vectorx_t p(frame_tracking_[settings_.tracking_frames_[i]]->GetParameterSize());
    // p << q, v, settings_.tracking_kp[i], settings_.tracking_kd[i], q_des, v_des, settings_.tracking_weights_[i];
    //
    // matrixx_t Jad, GNad;
    // frame_tracking_[settings_.tracking_frames_[i]]->GetGaussNewton(x_zero, p, Jad, GNad);
    // GNad *= 2;
    //
    // vectorx_t yad;
    // frame_tracking_[settings_.tracking_frames_[i]]->GetFunctionValue(x_zero, p, yad);
    // vectorx_t lin_term = 2*Jad.transpose()*yad;
    //
    // matrixx_t Had;
    // frame_tracking_[settings_.tracking_frames_[i]]->GetHessian(x_zero, p, 2*yad, Had);
    // Had += 2*Jad.transpose()*Jad;

    // // TODO: Remove after debugging
    // g += lin_term;
    // P += Had;

    // std::cout << "analytic lin term: " <<
    //     (2*J.topRows<3>().transpose()*settings_.tracking_weights_.at(i).asDiagonal()*(frame_acc_terms - a_control)).transpose() << std::endl;
    // std::cout << "ad lin term: " << lin_term.transpose() << std::endl;
    //
    // std::cout << "analytic hessian:\n" << 2*J.topRows<3>().transpose()*settings_.tracking_weights_.at(i).asDiagonal()*J.topRows<3>() << std::endl;
    // std::cout << "ad hessian:\n" << Had << std::endl;
    // std::cout << "ad GN:\n" << GNad << std::endl;
    //
    // std::cout << "analytic J:\n" << J << std::endl;
    // std::cout << "ad jac:\n" << Jad << std::endl;
    // std::cout << "ad y: " << yad.transpose() << std::endl;
    // std::cout << "analytic y: " << (settings_.tracking_weights_[i].asDiagonal()*(frame_acc_terms - a_control)).transpose() << std::endl;

    // for (const auto& frame : settings_.tracking_frames_) {
    //     frame_tracking_.emplace(frame, std::make_unique<ad::CppADInterface>(
    //         std::bind(&WbcController::FrameTrackingFunction, this, frame, std::placeholders::_1, std::placeholders::_2,
    //             std::placeholders::_3),
    //             frame + "_wbc_frame_tracking",
    //             deriv_lib_path,
    //             ad::DerivativeOrder::SecondOrder, nv_, 2*nq_ + 2*nv_ + 2*3 + 3,
    //             true //compile_derivs
    //         ));
    // }
}
