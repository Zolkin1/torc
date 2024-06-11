//
// Created by zolkin on 6/10/24.
//

#include <cstdlib>
#include <hpipm_d_ocp_qp_utils.h>
#include <hpipm_timing.h>

#include "hpipm.h"

// TODO: Write tests

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

    SolverStatus HPIPM::Solve(HPIPMData& data, std::vector<vectorx_t>& u, std::vector<vectorx_t>& x, double& time) {

        // Timer -- including setup, solving, and parsing
        hpipm_timer timer;
        hpipm_tic(&timer);

        // Set all the data
        // Dynamics
        std::vector<double*> AA(qp_size_.nodes, nullptr);
        std::vector<double*> BB(qp_size_.nodes, nullptr);
        std::vector<double*> CC(qp_size_.nodes, nullptr);

        // TODO: Deal with ic

        for (int i = 1; i < qp_size_.nodes; i++) {
            AA.at(i) = data.Ak.at(i).data();
            BB.at(i) = data.Bk.at(i).data();
            CC.at(i) = data.Ck.at(i).data();
        }

        // Cost
        std::vector<double*> QQ(qp_size_.nodes, nullptr);
        std::vector<double*> RR(qp_size_.nodes, nullptr);
        std::vector<double*> SS(qp_size_.nodes, nullptr);
        std::vector<double*> qq(qp_size_.nodes, nullptr);
        std::vector<double*> rr(qp_size_.nodes, nullptr);

        for (int i = 1; i < qp_size_.nodes; i++) {
            QQ.at(i) = data.Qk.at(i).data();
            RR.at(i) = data.Rk.at(i).data();
            SS.at(i) = data.Sk.at(i).data();
            qq.at(i) = data.qk.at(i).data();
            rr.at(i) = data.rk.at(i).data();
        }

        // Constraints
        std::vector<double*> DD(qp_size_.nodes, nullptr);
        std::vector<double*> GG(qp_size_.nodes, nullptr);
        std::vector<double*> uub(qp_size_.nodes, nullptr);
        std::vector<double*> llb(qp_size_.nodes, nullptr);

        for (int i = 1; i < qp_size_.nodes; i++) {
            DD.at(i) = data.Dk.at(i).data();
            GG.at(i) = data.Gk.at(i).data();
            uub.at(i) = data.uub.at(i).data();
            llb.at(i) = data.ulb.at(i).data();
        }

        // TODO: Check these
        int** hidxbx = nullptr;
        double** hlbx = nullptr;
        double** hubx = nullptr;
        int** hidxbu = nullptr;
        double** hlbu = nullptr;
        double** hubu = nullptr;
        double** hZl = nullptr;
        double** hZu = nullptr;
        double** hzl = nullptr;
        double** hzu = nullptr;
        int** hidxs = nullptr;
        double** hlls = nullptr;
        double** hlus = nullptr;

        d_ocp_qp_set_all(AA.data(), BB.data(), CC.data(), QQ.data(),
                         SS.data(), RR.data(), qq.data(), rr.data(),
                         hidxbx, hlbx, hubx,
                         hidxbu, hlbu, hubu, GG.data(),
                         DD.data(), llb.data(), uub.data(), hZl, hZu,
                         hzl, hzu, hidxs, hlls, hlus, &qp_);

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