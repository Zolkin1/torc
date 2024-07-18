//
// Created by zolkin on 6/10/24.
//

#ifndef TORC_IPOPT_H
#define TORC_IPOPT_H

#include "ipopt_interface.h"
#include "solver_status.h"

namespace torc::solvers {
    struct IPOPTSettings {
        double tol;
        double max_iter;
        double max_time;
        double dual_inf_tol;
        std::filesystem::path output_file;

        bool operator==(const IPOPTSettings& other) const {
            return (tol == other.tol) && (max_iter == other.max_iter)
                && (max_time == other.max_time) && (dual_inf_tol == other.dual_inf_tol)
                && (output_file == other.output_file);
        }

        bool operator!=(const IPOPTSettings& other) const {
            return !((tol == other.tol) && (max_iter == other.max_iter)
                   && (max_time == other.max_time) && (dual_inf_tol == other.dual_inf_tol)
                   && (output_file == other.output_file));
        }
    };

    class IPOPT {
    public:
        // TODO: Support different forms of logging (i.e. not just printing to cout)
        IPOPT();

        IPOPT(const IPOPTSettings& settings);

        IPOPT(const std::filesystem::path& settings_file);

        // TODO: What should this interface look like?
        //  Want to allow an MPCbase object and generic optimization problems be solved
        //  Should have one interface that takes in a cost function and a vector of constraints
        //  The cost function will be of one type and can be templated. The vector of constraints can in theory
        //  be of different types, but they will need to all be able to provide a derivative.
        //  Then there can be a second interface that will take in some MPCbase type object and just solve that.
        //  In both cases and initial guess can be provided, and if one is not provided, then we will guess all zeros
        SolverStatus SolveNLP();

        void UpdateSettings(const IPOPTSettings& settings);

        void UpdateSettings(const std::filesystem::path& settings_file);

    protected:
        bool Initialize();

        void AssignSettings();

        IPOPTSettings settings_;
        std::filesystem::path settings_file_;

        Ipopt::SmartPtr<Ipopt::IpoptApplication> app_;

    private:
    };
} // torc::solvers


#endif //TORC_IPOPT_H
