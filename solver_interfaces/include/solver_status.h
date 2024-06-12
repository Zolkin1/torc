//
// Created by zolkin on 6/10/24.
//

#ifndef TORC_SOLVER_STATUS_H
#define TORC_SOLVER_STATUS_H

namespace torc::solvers {
    enum SolverStatus {
        Solved,
        SolvedLowTol,
        Infeasible,
        MaxIters,
        TimeLimit,
        InvalidSetting,
        InvalidData,
        InitializationFailed,
        Ok,
        Unsolved,
        Error
    };

    void PrintStatus(std::ostream& out, const SolverStatus& status);
} // torc::solvers

#endif //TORC_SOLVER_STATUS_H
