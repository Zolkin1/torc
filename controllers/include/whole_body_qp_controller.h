//
// Created by zolkin on 7/17/24.
//

#ifndef TORC_WHOLE_BODY_QP_CONTROLLER_H
#define TORC_WHOLE_BODY_QP_CONTROLLER_H

#include <filesystem>
#include <Eigen/Core>

#include "full_order_rigid_body.h"
#include "osqp_interface.h"

namespace torc::controllers {
    using vectorx_t = Eigen::VectorXd;

    namespace fs = std::filesystem;
    class WholeBodyQPController {
    public:
        WholeBodyQPController(const std::string& name);

        WholeBodyQPController(const std::string& name, const models::FullOrderRigidBody& model);

        WholeBodyQPController(const std::string& name, const std::filesystem::path& urdf);

        WholeBodyQPController(const std::string& name, const std::filesystem::path& urdf,
                              const fs::path& config_file_path);

        void UpdateConfigFile(const fs::path& config_file_path);

        /**
         * @brief Computes the control action given the current state and contacts.
         *
         * @param state the current state of the robot. A vector (q, v).
         * @param contact the contacts to enforce as holonomic constraints
         * @return a vector of motor torques
         */
        vectorx_t ComputeControl(const vectorx_t& state, const models::Contact& contact);

        void UpdateTargetState(const vectorx_t& target_state);

    protected:
        /**
         * @brief parses the file given by config_file_path_ and assigns the gains/settings.
         */
        void ParseUpdateSettings();

        // ---------------------------------------- //
        // --------- Constraint Functions --------- //
        // ---------------------------------------- //
        void AddDynamicsConstraints(const vectorx_t& v, const models::Contact& contact);

        void AddHolonomicConstraints(const vectorx_t& state, const models::Contact& contact);

        void AddTorqueConstraints(const vectorx_t& v, const models::Contact& contact);

        void AddFrictionConeConstraints(const models::Contact& contact);

        void AddPositiveGRFConstraints(const models::Contact& contact);

        // -------- Member Variables -------- //
        fs::path config_file_path_;
        std::string name_;

        vectorx_t target_state_;

        solvers::OSQPInterface qp_solver_;

        std::unique_ptr<models::FullOrderRigidBody> model_;
    private:
    };
}


#endif //TORC_WHOLE_BODY_QP_CONTROLLER_H
