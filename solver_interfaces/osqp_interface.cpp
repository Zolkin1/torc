//
// Created by zolkin on 6/10/24.
//

#include "osqp_interface.h"

namespace torc::solvers {
    OSQPInterfaceSettings::OSQPInterfaceSettings() {
        rel_tol = -1;
        abs_tol = -1;
        verbose = true;
        polish = true;
        rho = -1;
        alpha = -1;
        adaptive_rho = true;
        max_iter = -1;
        max_time = -1;
    }

    OSQPInterface::OSQPInterface()
        : settings_() {
        SetSettings();
    }

    OSQPInterface::OSQPInterface(const torc::solvers::OSQPInterfaceSettings& settings)
        : settings_(settings) {
        SetSettings();
    }

    SolverStatus OSQPInterface::ResetData(const torc::solvers::sp_matrixx_t& P, vectorx_t& w,
                                          constraints::SparseBoxConstraints& constraints) {
        qp_solver_.data()->clearLinearConstraintsMatrix();
        qp_solver_.data()->clearHessianMatrix();
        qp_solver_.clearSolver();

        qp_solver_.data()->setNumberOfVariables(w.size());
        qp_solver_.data()->setNumberOfConstraints(constraints.A.rows());

        // Set constraints
        if (!(qp_solver_.data()->setLinearConstraintsMatrix(constraints.A) &&
              qp_solver_.data()->setBounds(constraints.lb, constraints.ub))) {
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
                                           const constraints::SparseBoxConstraints& constraints) {
        // Constraints
        if (!(qp_solver_.updateLinearConstraintsMatrix(constraints.A) &&
              qp_solver_.updateBounds(constraints.lb, constraints.ub))) {
            return InvalidData;
        }

        // Set solver costs
        if (!(qp_solver_.updateHessianMatrix(P) && qp_solver_.updateGradient(w))) {
            return InvalidData;
        }

        return Ok;
    }

    void OSQPInterface::UpdateSettings(const torc::solvers::OSQPInterfaceSettings& settings) {
        settings_ = settings;
        SetSettings();
    }

    void OSQPInterface::SetSettings() {
        if (settings_.rel_tol > 0) {
            qp_solver_.settings()->setRelativeTolerance(settings_.rel_tol);
        }

        if (settings_.abs_tol > 0) {
            qp_solver_.settings()->setAbsoluteTolerance(settings_.abs_tol);
        }

        qp_solver_.settings()->setVerbosity(settings_.verbose);

        qp_solver_.settings()->setPolish(settings_.polish);

        qp_solver_.settings()->setAdaptiveRho(settings_.adaptive_rho);

        if (settings_.rho > 0) {
            qp_solver_.settings()->setRho(settings_.rho);
        }

        if (settings_.alpha > 0) {
            qp_solver_.settings()->setAlpha(settings_.alpha);
        }

        if (settings_.max_iter > 0) {
            qp_solver_.settings()->setMaxIteration(settings_.max_iter);
        }

        // TODO: allow for max time
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