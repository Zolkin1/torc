//
// Created by zolkin on 6/10/24.
//

#include "ipopt.h"

namespace torc::solvers {
    IPOPT::IPOPT() {
        app_ = IpoptApplicationFactory();
    }

    IPOPT::IPOPT(const torc::solvers::IPOPTSettings& settings)
        : IPOPT() {
            settings_ = settings;
            AssignSettings();
    }

    IPOPT::IPOPT(const std::filesystem::path& settings_file)
            : IPOPT() {
        settings_file_ = settings_file;
        app_->Options()->SetStringValue("option_file_name", settings_file.string());
    }

    SolverStatus IPOPT::SolveNLP() {
        // Initialize the IpoptApplication and process the options
        if (!Initialize()) {
            return InitializationFailed;
        }

        // Solve the problem
        Ipopt::SmartPtr<Ipopt::TNLP> nlp = new IPOPTInterface();

        Ipopt::ApplicationReturnStatus status;
        status = app_->OptimizeTNLP(nlp);

        if(status == Ipopt::Solve_Succeeded )
        {
            return Solved;
        } else if (status == Ipopt::Solved_To_Acceptable_Level) {
            return SolvedLowTol;
        } else if (status == Ipopt::Infeasible_Problem_Detected) {
            return Infeasible;
        } else if (status == Ipopt::Maximum_CpuTime_Exceeded || status == Ipopt::Maximum_WallTime_Exceeded) {
            return TimeLimit;
        } else if (status == Ipopt::Maximum_Iterations_Exceeded) {
            return MaxIters;
        } else if (status == Ipopt::Invalid_Option) {
            return InvalidSetting;
        } else {
            return Error;
        }
    }

    void IPOPT::UpdateSettings(const torc::solvers::IPOPTSettings& settings) {
        if (settings_ != settings) {
            settings_ = settings;
            AssignSettings();
        }
    }

    void IPOPT::UpdateSettings(const std::filesystem::path& settings_file) {
        if (settings_file_ != settings_file) {
            settings_file_ = settings_file;
            app_->Options()->SetStringValue("option_file_name", settings_file.string());
        }
    }

    bool IPOPT::Initialize() {
        Ipopt::ApplicationReturnStatus status;
        status = app_->Initialize();
        if( status != Ipopt::Solve_Succeeded )
        {
            std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
            std::cout << "status: " << (int) status << std::endl;
            return false;
        }

        return true;
    }

    void IPOPT::AssignSettings() {
        if (settings_.tol > 0) {
            app_->Options()->SetNumericValue("tol", settings_.tol);
        }

        if (settings_.max_iter > 0) {
            app_->Options()->SetNumericValue("max_iter", settings_.max_iter);
        }

        if (settings_.max_time > 0) {
            app_->Options()->SetNumericValue("max_time", settings_.max_time);
        }

        if (settings_.dual_inf_tol > 0) {
            app_->Options()->SetNumericValue("dual_inf_tol", settings_.dual_inf_tol);
        }

        if (std::filesystem::exists(settings_.output_file)) {
            app_->Options()->SetStringValue("output_file", settings_.output_file);
        }
    }
} // torc::solvers