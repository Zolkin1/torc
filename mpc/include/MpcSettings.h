//
// Created by zolkin on 1/18/25.
//

#ifndef MPCSETTINGS_H
#define MPCSETTINGS_H

#include <vector>
#include <filesystem>
#include "hpipm-cpp/hpipm-cpp.hpp"
#include "cost_function.h"
#include "CollisionData.h"

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
        void ParseTargetSettings();

        void Print();

        // Model
        std::vector<std::string> joint_skip_names;
        std::vector<double> joint_skip_values;

        // General
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

        // Constraints
        double friction_coef;
        double max_grf;
        double min_grf;
        double friction_margin;
        double polytope_delta;
        double polytope_shrinking_rad;
        int swing_start_node;
        int swing_end_node;
        int holonomic_start_node;
        int holonomic_end_node;
        int collision_start_node;
        int collision_end_node;
        int polytope_start_node;
        int polytope_end_node;
        double swing_buffer;    // TODO: Fill in
        std::vector<CollisionData> collision_data;
        std::vector<std::string> polytope_frames;
        std::vector<std::pair<std::string, std::string>> poly_contact_pairs; // Poly frame first, contact frame second

        std::vector<CostData> cost_data;

        // Contact Info
        std::vector<std::string> contact_frames;
        int num_contact_locations;
        std::vector<double> hip_offsets;

        // Swing Info
        double apex_height;
        double apex_time;   // 0-1
        double default_ground_height;

        // QP Settings
        hpipm::OcpQpIpmSolverSettings qp_settings;

        // Line Search
        double ls_eta;
        double ls_alpha_min;
        double ls_theta_max;
        double ls_theta_min;
        double ls_gamma_theta;
        double ls_gamma_alpha;
        double ls_gamma_phi;

        // Targets
        vectorx_t q_target;
        vectorx_t v_target;

        // Log
        bool log;

    protected:
    private:
        fs::path config_file_;
    };
}


#endif //MPCSETTINGS_H
