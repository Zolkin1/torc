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

    bool IPOPT::SolveNLP() {
        // Initialize the IpoptApplication and process the options
        Initialize();

        // Ask Ipopt to solve the problem
        Ipopt::SmartPtr<Ipopt::TNLP> nlp = new IPOPTInterface();

        Ipopt::ApplicationReturnStatus status;
        status = app_->OptimizeTNLP(nlp);

        if( status == Ipopt::Solve_Succeeded )
        {
            std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
        }
        else
        {
            std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
        }

        // As the SmartPtrs go out of scope, the reference count
        // will be decremented and the objects will automatically
        // be deleted.

        std::cout << "status: " << (int) status << std::endl;
        return (int) status;
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
        }
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