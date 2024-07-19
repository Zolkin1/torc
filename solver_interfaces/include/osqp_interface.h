//
// Created by zolkin on 6/10/24.
//

#ifndef TORC_OSQP_INTERFACE_H
#define TORC_OSQP_INTERFACE_H

#include "OsqpEigen/OsqpEigen.h"

#include "solver_status.h"
#include "constraint.h"

namespace torc::solvers {
    using vectorx_t = Eigen::VectorXd;
    using matrixx_t = Eigen::MatrixXd;

    using sp_matrixx_t = Eigen::SparseMatrix<double>;

    struct OSQPInterfaceSettings {
        double rel_tol;
        double abs_tol;
        bool verbose;
        bool polish;
        double rho;
        double alpha;
        bool adaptive_rho;
        long max_iter;
        double max_time;

        OSQPInterfaceSettings();
    };

    class OSQPInterface {
    public:
        OSQPInterface();

        explicit OSQPInterface(const OSQPInterfaceSettings& settings);

        // TODO: Figure out how to make the vectors const (issues with OSQPEigen I think)
        SolverStatus ResetData(const sp_matrixx_t& P, vectorx_t& w, constraints::SparseBoxConstraints& constraints);

        SolverStatus UpdateData(const sp_matrixx_t& P, const vectorx_t& w,
                                const constraints::SparseBoxConstraints& constraints);

        void UpdateSettings(const OSQPInterfaceSettings& settings);

        SolverStatus Solve(vectorx_t& sol);

    protected:
        void SetSettings();

        OsqpEigen::Solver qp_solver_;

        OSQPInterfaceSettings settings_;
    private:
    };
} // torc::solvers


#endif //TORC_OSQP_INTERFACE_H
