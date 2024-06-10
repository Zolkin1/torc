//
// Created by zolkin on 6/10/24.
//
#include <iostream>

#include "solver_status.h"

namespace torc::solvers {
    void PrintStatus(std::ostream& out, const SolverStatus& status) {
        switch (status) {
            case Solved:
                out << "Solved";
                break;
            case SolvedLowTol:
                out << "Solved Low Tolerance";
                break;
            case Infeasible:
                out << "Infeasible";
                break;
            case TimeLimit:
                out << "Time Limit";
                break;
            case MaxIters:
                out << "Maximum Iterations";
                break;
            case InvalidSetting:
                out << "Invalid settings";
                break;
            default:
                out << "Error";
                break;
        }
    }
}