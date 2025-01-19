//
// Created by zolkin on 1/18/25.
//

#ifndef MPCSETTINGS_H
#define MPCSETTINGS_H

#include <vector>
#include <filesystem>

#include "cost_function.h"

namespace torc::mpc {
    namespace fs = std::filesystem;

    class MpcSettings {
    public:
        MpcSettings(const fs::path& config_file);

        void ParseConfigFile(const fs::path& config_file);

        void ParseJointDefaults();
        void ParseGeneralSettings();
        void ParseSolverSettings();
        void ParseConstraintSettings();
        void ParseCostSettings();
        void ParseContactSettings();
        void ParseLineSearchSettings();

        void Print();

        std::vector<std::string> joint_skip_names;
        std::vector<double> joint_skip_values;

        int nodes;
        bool verbose;
        std::vector<double> dt;
        bool compile_derivs;
        fs::path deriv_lib_path;
        std::string base_frame;
        bool scale_cost;
        int max_initial_solves;
        double initial_solve_tolerance;
        int nodes_full_dynamics;
        double terminal_weight;

        double friction_coef;
        double max_grf;
        double friction_margin;
        std::vector<std::pair<std::string, std::string>> collision_frames;
        std::vector<std::pair<double, double>> collision_radii;

        std::vector<CostData> cost_data;

        std::vector<std::string> contact_frames;
        int num_contact_locations;
        std::vector<double> hip_offsets;

        double ls_eta;
        double ls_alpha_min;
        double ls_theta_max;
        double ls_theta_min;
        double ls_gamma_theta;
        double ls_gamma_alpha;
        double ls_gamma_phi;

    protected:
    private:
        fs::path config_file_;
    };
}


#endif //MPCSETTINGS_H
