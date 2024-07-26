//
// Created by zolkin on 6/12/24.
//

#ifndef TORC_CLARABEL_INTERFACE_H
#define TORC_CLARABEL_INTERFACE_H

#include <Clarabel>

#include "solver_status.h"

namespace torc::solvers {
    using sp_matrixx_t = Eigen::SparseMatrix<double, Eigen::ColMajor>;
    using matrixx_t = Eigen::MatrixXd;
    using vectorx_t = Eigen::VectorXd;

    struct ClarabelInterfaceSettings {
        bool verbose;
        int max_iters;
        double time_limit;
        double tol_feas;
        double tol_gap_abs;
        double tol_gap_rel;

        ClarabelInterfaceSettings();

        bool operator==(const ClarabelInterfaceSettings& other) const;
        bool operator!=(const ClarabelInterfaceSettings& other) const;
    };

    // TODO: Remove when we have this defined with the constraints manager
    enum Constraints {
        Equality,
        UpperBound,
        LowerBound,
        UpperLowerBound
    };

    struct ClarabelData {
        std::vector<std::pair<Constraints, int>> constraint_data;
        sp_matrixx_t Q;
        vectorx_t q;
        sp_matrixx_t A;
        vectorx_t b;

        bool ConsistencyCheck();
    };

    class ClarabelInterface {
    public:
        ClarabelInterface();

        explicit ClarabelInterface(const ClarabelInterfaceSettings& settings);

        SolverStatus SetData(ClarabelData& data);

        SolverStatus Solve(vectorx_t& sol);

        SolverStatus Solve(vectorx_t& sol, vectorx_t& dual);

        SolverStatus Solve(vectorx_t& sol, vectorx_t& dual, vectorx_t& slacks);

        void UpdateSettings(const ClarabelInterfaceSettings& settings);

        [[nodiscard]] ClarabelInterfaceSettings GetSettings() const;

    protected:
        void SetSettings();

        SolverStatus ParseClarabelStatus(const clarabel::SolverStatus& status);

    private:
        std::unique_ptr<clarabel::DefaultSolver<double>> solver_;
        clarabel::DefaultSettings<double> clara_settings_;
        std::vector<clarabel::SupportedConeT<double>> cones_;

        ClarabelInterfaceSettings settings_;
        ClarabelData data_;
    };
} // torc::solvers


#endif //TORC_CLARABEL_INTERFACE_H
