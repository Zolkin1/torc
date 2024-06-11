//
// Created by zolkin on 6/10/24.
//

#include <cstdlib>
#include <hpipm_d_ocp_qp_utils.h>
#include <hpipm_timing.h>

#include "hpipm.h"

namespace torc::solvers {
    bool operator==(const HPIPMQPSize& size1, const HPIPMQPSize& size2) {
         return (size1.num_states == size2.num_states) && (size1.nodes == size2.nodes)
            && (size1.num_inputs == size2.num_inputs)
            && (size1.num_inp_box_constraints == size2.num_inp_box_constraints)
            && (size1.num_state_box_constraints == size2.num_state_box_constraints)
            && (size1.num_ineq_constraints == size2.num_ineq_constraints)
            && (size1.num_inp_slack_box_constraints == size2.num_inp_slack_box_constraints)
            && (size1.num_state_slack_box_constraints == size2.num_state_slack_box_constraints)
            && (size1.num_ineq_slacks == size2.num_ineq_slacks);
    }

    bool operator!=(const HPIPMQPSize& size1, const HPIPMQPSize& size2) {
        return !(size1 == size2);
    }

    bool operator==(const HPIPMSettings& s1, const HPIPMSettings& s2) {
        return (s1.mu0 == s2.mu0) && (s1.max_iter == s2.max_iter)
               && (s1.warm_start == s2.warm_start)
               && (s1.sqr_root_alg == s2.sqr_root_alg)
               && (s1.mode == s2.mode)
               && (s1.pred_corr == s2.pred_corr)
               && (s1.cond_pred_corr == s2.cond_pred_corr);
    }

    bool operator!=(const HPIPMSettings& s1, const HPIPMSettings& s2) {
        return !(s1 == s2);
    }

    HPIPM::HPIPM(torc::solvers::HPIPMQPSize qp_size)
        : qp_size_(std::move(qp_size)), settings_() {
        Init(qp_size);
    }

    void HPIPM::Init(const torc::solvers::HPIPMQPSize& qp_size) {
        if (qp_size != qp_size_) {
            qp_size_ = qp_size;
            AllocateMemory();
        }
    }

    void HPIPM::UpdateSettings(const torc::solvers::HPIPMSettings& settings) {
        if (settings != settings_) {
            settings_ = settings;

            // Set default settings
            d_ocp_qp_ipm_arg_set_default(settings_.mode, &arg_);
            if (settings_.max_iter > 0) {
                d_ocp_qp_ipm_arg_set_iter_max(&settings_.max_iter, &arg_);
            }

            if (settings_.mu0 > 0) {
                d_ocp_qp_ipm_arg_set_mu0(&settings_.mu0, &arg_);
            }

            if (settings_.sqr_root_alg != RicDefault) {
                int alg = static_cast<int>(settings_.sqr_root_alg);
                d_ocp_qp_ipm_arg_set_ric_alg(&alg, &arg_);
            }

            if (settings_.warm_start != WSDefault) {
                int ws = static_cast<int>(settings_.warm_start);
                d_ocp_qp_ipm_arg_set_warm_start(&ws, &arg_);
            }

            if (settings_.pred_corr) {
                int pred_corr = 1;
                d_ocp_qp_ipm_arg_set_pred_corr(&pred_corr, &arg_);
            } else {
                int pred_corr = 0;
                d_ocp_qp_ipm_arg_set_pred_corr(&pred_corr, &arg_);
            }

            if (settings_.alpha_min > 0) {
                d_ocp_qp_ipm_arg_set_alpha_min(&settings_.alpha_min, &arg_);
            }

            if (settings_.stat_tol > 0) {
                d_ocp_qp_ipm_arg_set_tol_stat(&settings_.stat_tol, &arg_);
            }

            if (settings_.eq_tol > 0) {
                d_ocp_qp_ipm_arg_set_tol_eq(&settings_.eq_tol, &arg_);
            }

            if (settings_.ineq_tol > 0) {
                d_ocp_qp_ipm_arg_set_tol_ineq(&settings_.ineq_tol, &arg_);
            }

            if (settings_.comp_tol > 0) {
                d_ocp_qp_ipm_arg_set_tol_comp(&settings_.comp_tol, &arg_);
            }

            if (settings_.reg_prim > 0) {
                d_ocp_qp_ipm_arg_set_reg_prim(&settings_.reg_prim, &arg_);
            }
        }
    }

    void HPIPM::AllocateMemory() {
        // ------- Create the dimensions ------- //
        const hpipm_size_t dim_size = d_ocp_qp_dim_memsize(qp_size_.nodes);

        if (dim_mem_) { free(dim_mem_); }
        dim_mem_ = malloc(dim_size);
        if (dim_mem_ == nullptr) {throw std::bad_alloc();}

        d_ocp_qp_dim_create(qp_size_.nodes, &dim_, dim_mem_);
        d_ocp_qp_dim_set_all(qp_size_.num_states.data(), qp_size_.num_inputs.data(),
                             qp_size_.num_state_box_constraints.data(),
                             qp_size_.num_inp_box_constraints.data(),
                             qp_size_.num_ineq_constraints.data(),
                             qp_size_.num_state_slack_box_constraints.data(),
                             qp_size_.num_inp_slack_box_constraints.data(),
                             qp_size_.num_ineq_slacks.data(), &dim_);

        // ------- Create the QP ------- //
        const hpipm_size_t size = d_ocp_qp_memsize(&dim_);
        if (qp_mem_) { free(qp_mem_); }
        qp_mem_ = malloc(size);
        if (qp_mem_ == nullptr) {throw std::bad_alloc();}

        d_ocp_qp_create(&dim_, &qp_, qp_mem_);


        // ------- Create the Solution Space ------- //
        hpipm_size_t qp_sol_size = d_ocp_qp_sol_memsize(&dim_);
        if (qp_sol_mem_) { free(qp_sol_mem_); }
        qp_sol_mem_ = malloc(qp_sol_size);
        if (qp_sol_mem_ == nullptr) {throw std::bad_alloc();}

        d_ocp_qp_sol_create(&dim_, &qp_sol_, qp_sol_mem_);

        // -------- IPM Arguments -------- //
        hpipm_size_t ipm_arg_size = d_ocp_qp_ipm_arg_memsize(&dim_);
        if (ipm_arg_mem_) { free(ipm_arg_mem_); }
        ipm_arg_mem_ = malloc(ipm_arg_size);
        if (ipm_arg_mem_ == nullptr) {throw std::bad_alloc();}

        d_ocp_qp_ipm_arg_create(&dim_, &arg_, ipm_arg_mem_);

        // -------- IPM Workspace -------- //
        hpipm_size_t ipm_size = d_ocp_qp_ipm_ws_memsize(&dim_, &arg_);
        if (ipm_mem_) { free(ipm_mem_); }
        ipm_mem_ = malloc(ipm_size);
        if (ipm_mem_ == nullptr) {throw std::bad_alloc();}

        d_ocp_qp_ipm_ws_create(&dim_, &arg_, &workspace_, ipm_mem_);

        const hpipm_mode mode_default = BALANCE;
        d_ocp_qp_ipm_arg_set_default(mode_default, &arg_);
    }

    SolverStatus HPIPM::Solve(std::vector<vectorx_t>& u, std::vector<vectorx_t>& x, double& time) {

        // Timer -- including setup, solving, and parsing
        hpipm_timer timer;
        hpipm_tic(&timer);

        // Set all the data
//        d_ocp_qp_set_all(hA, hB, hb, hQ,
//                         hS, hR, hq, hr,
//                         hidxbx, hlbx, hubx,
//                         hidxbu, hlbu, hubu, hC,
//                         hD, hlg, hug, hZl, hZu,
//                         hzl, hzu, hidxs, hlls, hlus, &qp);

        // Solve
        d_ocp_qp_ipm_solve(&qp_, &qp_sol_, &arg_, &workspace_);

        //Status
        int hpipm_status;
        d_ocp_qp_ipm_get_status(&workspace_, &hpipm_status);

        ParseSolution(u, x);

        // Get time
        time = hpipm_toc(&timer);

        // Parse return status
        switch (hpipm_status) {
            case SUCCESS:
                return Solved;
            case MAX_ITER:
                return MaxIters;
            case MIN_STEP:
                return Infeasible; // TODO: I don't think this is infeasible
            case NAN_SOL:
                return Error;
            case INCONS_EQ:
                return Infeasible;
            default:
                return Error;
        }
    }

    void HPIPM::ParseSolution(std::vector<vectorx_t>& u, std::vector<vectorx_t>& x) {
        u.resize(qp_size_.nodes-1);
        x.resize(qp_size_.nodes);

        for (int i = 0; i < qp_size_.nodes; i++) {
            if (i < qp_size_.nodes - 1) {
                u.at(i).resize(qp_size_.num_inputs.at(i));
                d_ocp_qp_sol_get_u(i, &qp_sol_, u.at(i).data());
            }

            x.at(i).resize(qp_size_.num_states.at(i));
            d_ocp_qp_sol_get_u(i, &qp_sol_, x.at(i).data());
        }
    }

    HPIPM::~HPIPM() {
        if (dim_mem_) { free(dim_mem_); }
        if (qp_mem_) { free(qp_mem_); }
        if (qp_sol_mem_) { free(qp_sol_mem_); }
        if (ipm_arg_mem_) { free(ipm_arg_mem_); }
        if (ipm_mem_) { free(ipm_mem_); }
    }

} // torc::solvers