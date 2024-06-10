//
// Created by zolkin on 6/10/24.
//

#include "osqp_interface.h"

namespace torc::solvers {
    SolverStatus OSQPInterface::ResetData(const torc::solvers::sp_matrixx_t& P, vectorx_t& w,
                                          const torc::solvers::sp_matrixx_t& A, torc::solvers::vectorx_t& lb,
                                          torc::solvers::vectorx_t& ub) {
        qp_solver_.data()->clearLinearConstraintsMatrix();
        qp_solver_.data()->clearHessianMatrix();
        qp_solver_.clearSolver();

        qp_solver_.data()->setNumberOfVariables(w.size());
        qp_solver_.data()->setNumberOfConstraints(A.rows());

        // Set constraints
        if (!(qp_solver_.data()->setLinearConstraintsMatrix(A) &&
              qp_solver_.data()->setBounds(lb, ub))) {
            return InvalidData;
        }

        // Set solver costs
        if (!(qp_solver_.data()->setHessianMatrix(P) && qp_solver_.data()->setGradient(w))) {
            return InvalidData;
        }

        // Initialize
        if (!qp_solver_.initSolver()) {
            return InitializationFailed;
        }

        return Ok;
    }

    // Note: this one will only work if sparsity pattern did not change.
    SolverStatus OSQPInterface::UpdateData(const torc::solvers::sp_matrixx_t& P, const torc::solvers::vectorx_t& w,
                                           const torc::solvers::sp_matrixx_t& A, const torc::solvers::vectorx_t& lb,
                                           const torc::solvers::vectorx_t& ub) {
        // Constraints
        if (!(qp_solver_.updateLinearConstraintsMatrix(A) &&
              qp_solver_.updateBounds(lb, ub))) {
            return InvalidData;
        }

        // Set solver costs
        if (!(qp_solver_.updateHessianMatrix(P) && qp_solver_.updateGradient(w))) {
            return InvalidData;
        }

        return Ok;
    }

    SolverStatus OSQPInterface::Solve(torc::solvers::vectorx_t& sol) {
        const OsqpEigen::ErrorExitFlag error_flag = qp_solver_.solveProblem();

        if (error_flag != OsqpEigen::ErrorExitFlag::NoError) {
            return Error;
        }

        const OsqpEigen::Status status = qp_solver_.getStatus();
        sol = qp_solver_.getSolution();

        if (status == OsqpEigen::Status::DualInfeasible || status == OsqpEigen::Status::PrimalInfeasible
            || status == OsqpEigen::Status::DualInfeasibleInaccurate
            || status == OsqpEigen::Status::PrimalInfeasibleInaccurate) {
            return Infeasible;
        } else if (status == OsqpEigen::Status::SolvedInaccurate) {
            return SolvedLowTol;
        } else if (status == OsqpEigen::Status::MaxIterReached) {
            return MaxIters;
        } else if (status == OsqpEigen::Status::Solved) {
            return Solved;
        } else if (status == OsqpEigen::Status::NonCvx) {
            return InvalidData;
        } else {
            return Error;
        }
    }

} // torc::solvers