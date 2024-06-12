//
// Created by zolkin on 6/12/24.
//

#include "clarabel_interface.h"

namespace torc::solvers {

    ClarabelInterfaceSettings::ClarabelInterfaceSettings() {
        verbose = true;
        max_iters = -1;
        time_limit = -1;
        tol_feas = -1;
        tol_gap_abs = -1;
        tol_gap_rel = -1;
    }

    bool ClarabelInterfaceSettings::operator==(const torc::solvers::ClarabelInterfaceSettings& other) const {
        return (verbose == other.verbose) && (max_iters == other.max_iters)
            && (time_limit == other.time_limit) && (tol_feas == other.tol_feas)
            && (tol_gap_rel == other.tol_gap_rel) && (tol_gap_abs == other.tol_gap_abs);
    }

    bool ClarabelInterfaceSettings::operator!=(const torc::solvers::ClarabelInterfaceSettings& other) const {
        return !(*this == other);
    }

    bool ClarabelData::ConsistencyCheck() {
        int total_constraints = 0;
        for (const auto constraints : constraint_data) {
            total_constraints += constraints.second;
        }

        if (A.rows() != total_constraints || b.size() != total_constraints) {
            return false;
        }

        if (Q.rows() != Q.cols() || Q.cols() != q.size() || A.cols() != q.size()) {
            return false;
        }

        return true;
    }

    ClarabelInterface::ClarabelInterface()
        : settings_(), clara_settings_(clarabel::DefaultSettings<double>::default_settings()) {}

    ClarabelInterface::ClarabelInterface(const torc::solvers::ClarabelInterfaceSettings& settings)
            : settings_(settings), clara_settings_(clarabel::DefaultSettings<double>::default_settings()) {
        SetSettings();
    }

    void ClarabelInterface::UpdateSettings(const torc::solvers::ClarabelInterfaceSettings& settings) {
        settings_ = settings;
        SetSettings();
    }

    SolverStatus ClarabelInterface::SetData(ClarabelData& data) {
        if (!data.ConsistencyCheck()) {
            return InitializationFailed;
        }

        data_ = data;

        cones_ = {};
        for (const auto& constraint: data_.constraint_data) {
            switch (constraint.first) {
                case Equality:
                    cones_.push_back(clarabel::ZeroConeT<double>(constraint.second));
                    break;
                case UpperBound:
                    cones_.push_back(clarabel::NonnegativeConeT<double>(constraint.second));
                    break;
                case UpperLowerBound:
                case LowerBound:
                    // Only accept constraints in upper bound form
                    return InitializationFailed;
                default:
                    throw std::runtime_error("Invalid constraint type.");
            }
        }

        solver_ = std::make_unique<clarabel::DefaultSolver<double>>(data_.Q, data_.q,
                                                                    data_.A, data_.b,
                                                                    cones_, clara_settings_);


        return Ok;
    }

    SolverStatus ClarabelInterface::Solve(torc::solvers::vectorx_t& sol) {
        solver_->solve();

        clarabel::DefaultSolution<double> clara_sol = solver_->solution();

        sol = clara_sol.x;

        return ParseClarabelStatus(clara_sol.status);
    }

    SolverStatus ClarabelInterface::Solve(torc::solvers::vectorx_t& sol, vectorx_t& dual) {
        solver_->solve();

        clarabel::DefaultSolution<double> clara_sol = solver_->solution();

        sol = clara_sol.x;
        dual = clara_sol.z;

        return ParseClarabelStatus(clara_sol.status);
    }

    SolverStatus ClarabelInterface::Solve(torc::solvers::vectorx_t& sol, vectorx_t& dual, vectorx_t& slacks) {
        solver_->solve();

        clarabel::DefaultSolution<double> clara_sol = solver_->solution();

        sol = clara_sol.x;
        dual = clara_sol.z;
        slacks = clara_sol.s;

        return ParseClarabelStatus(clara_sol.status);
    }

    ClarabelInterfaceSettings ClarabelInterface::GetSettings() const {
        return settings_;
    }

    void ClarabelInterface::SetSettings() {
        clara_settings_ = clarabel::DefaultSettings<double>::default_settings();

        if (settings_.max_iters > 0) {
            clara_settings_.max_iter = settings_.max_iters;
        }

        clara_settings_.verbose = settings_.verbose;

        if (settings_.time_limit > 0) {
            clara_settings_.time_limit = settings_.time_limit;
        }

        if (settings_.tol_feas > 0) {
            clara_settings_.tol_feas = settings_.tol_feas;
        }

        if (settings_.tol_gap_rel > 0) {
            clara_settings_.tol_gap_rel = settings_.tol_gap_rel;
        }

        if (settings_.tol_gap_abs > 0) {
            clara_settings_.tol_gap_abs = settings_.tol_gap_abs;
        }

        // If we have a solver, we need to update the settings
        if (solver_) {
            solver_ = std::make_unique<clarabel::DefaultSolver<double>>(data_.Q, data_.q,
                                                                        data_.A, data_.b,
                                                                        cones_, clara_settings_);
        }
    }

    SolverStatus ClarabelInterface::ParseClarabelStatus(const clarabel::SolverStatus& status) {
        switch (status) {
            case clarabel::SolverStatus::Solved:
                return Solved;
            case clarabel::SolverStatus::Unsolved:
                return Unsolved;
            case clarabel::SolverStatus::PrimalInfeasible:
            case clarabel::SolverStatus::DualInfeasible:
                return Infeasible;
            case clarabel::SolverStatus::MaxIterations:
                return MaxIters;
            case clarabel::SolverStatus::AlmostSolved:
                return SolvedLowTol;
            case clarabel::SolverStatus::AlmostPrimalInfeasible:
            case clarabel::SolverStatus::AlmostDualInfeasible:
                return Infeasible;
            default:
                return Error;
        }
    }
} // torc::solvers
