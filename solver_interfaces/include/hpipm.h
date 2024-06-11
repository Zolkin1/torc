//
// Created by zolkin on 6/10/24.
//

#ifndef TORC_HPIPM_H
#define TORC_HPIPM_H

#include <eigen3/Eigen/Dense>
#include <hpipm_d_ocp_qp_sol.h>
#include <hpipm_d_ocp_qp_ipm.h>

extern "C" {
#include "hpipm_d_ocp_qp.h"
}

#include "solver_status.h"

namespace torc::solvers {
    using matrixx_t = Eigen::MatrixXd;
    using vectorx_t = Eigen::VectorXd;

    struct HPIPMQPSize {
        int nodes;
        std::vector<int> num_states;
        std::vector<int> num_inputs;
        std::vector<int> num_inp_box_constraints;
        std::vector<int> num_state_box_constraints;
        std::vector<int> num_ineq_constraints;
        std::vector<int> num_inp_slack_box_constraints;
        std::vector<int> num_state_slack_box_constraints;
        std::vector<int> num_ineq_slacks;

    };

    enum HPIPMWarmStart {
        NoWarmStart = 0,
        WarmStartPrimal,
        WarmStartPrimalDual,
        WSDefault
    };

    enum HPIPMSqrRootAlg {
        Riccati = 0,
        SqRtRiccati,
        RicDefault
    };

    struct HPIPMSettings {
        double mu0;
        int max_iter;
        HPIPMWarmStart warm_start;
        HPIPMSqrRootAlg sqr_root_alg;
        hpipm_mode mode;
        bool pred_corr;
        bool cond_pred_corr;
        double alpha_min;
        double eq_tol;
        double comp_tol;
        double ineq_tol;
        double stat_tol;
        double reg_prim;

        HPIPMSettings() {
            mu0 = 1e2;
            max_iter = 30;
            warm_start = NoWarmStart;
            sqr_root_alg = Riccati;
            mode = SPEED_ABS;
            pred_corr = true;
            cond_pred_corr = false;
            alpha_min = 1e-8;
            stat_tol = 1e-8;
            eq_tol = 1e-8;
            ineq_tol = 1e-8;
            comp_tol = 1e-8;
            reg_prim = 1e-12;
        }

    };

    struct HPIPMData {
        // Dynamics
        std::vector<matrixx_t> Ak;
        std::vector<matrixx_t> Bk;
        std::vector<vectorx_t> Ck;  // b in the HPIPM documentation
        vectorx_t ic;

        // Cost
        std::vector<matrixx_t> Rk;
        std::vector<matrixx_t> Qk;
        std::vector<matrixx_t> Sk;
        std::vector<vectorx_t> rk;
        std::vector<vectorx_t> qk;

        // Constraints
        std::vector<matrixx_t> Dk;
        std::vector<matrixx_t> Gk;  // C in the HPIPM documentation
        std::vector<vectorx_t> xub;
        std::vector<vectorx_t> xlb;
        std::vector<vectorx_t> uub;
        std::vector<vectorx_t> ulb;
    };

    bool operator==(const HPIPMQPSize& size1, const HPIPMQPSize& size2);
    bool operator!=(const HPIPMQPSize& size1, const HPIPMQPSize& size2);

    bool operator!=(const HPIPMSettings& s1, const HPIPMSettings& s2);
    bool operator==(const HPIPMSettings& s1, const HPIPMSettings& s2);


    class HPIPM {
    public:

        HPIPM(HPIPMQPSize qp_size);

        HPIPM(const HPIPMQPSize& qp_size, const HPIPMSettings& settings);

        void Init(const HPIPMQPSize& qp_size);

        void UpdateSettings(const HPIPMSettings& settings);

        SolverStatus Solve(HPIPMData& data, std::vector<vectorx_t>& u, std::vector<vectorx_t>& x, double& time);

        ~HPIPM();

    protected:
        void AllocateMemory();

        void ParseSolution(std::vector<vectorx_t>& u, std::vector<vectorx_t>& x);

        HPIPMQPSize qp_size_;

        HPIPMSettings settings_;

        void *dim_mem_;
        struct d_ocp_qp_dim dim_;

        void *qp_mem_;
        struct d_ocp_qp qp_;

        void* qp_sol_mem_;
        struct d_ocp_qp_sol qp_sol_;

        void *ipm_arg_mem_;
        struct d_ocp_qp_ipm_arg arg_;

        void *ipm_mem_;
        struct d_ocp_qp_ipm_ws workspace_;

    private:
    };
} // torc::solvers


#endif //TORC_HPIPM_H
