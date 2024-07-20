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
    using vector3_t = Eigen::Vector3d;
    using matrixx_t = Eigen::MatrixXd;

    namespace fs = std::filesystem;
    class WholeBodyQPController {
    public:
        WholeBodyQPController(const std::string& name);

        WholeBodyQPController(const std::string& name, const models::FullOrderRigidBody& model,
                              const fs::path& config_file_path);

        WholeBodyQPController(const std::string& name, const std::filesystem::path& urdf);

        WholeBodyQPController(const std::string& name, const std::filesystem::path& urdf,
                              const fs::path& config_file_path);

        void UpdateConfigFile(const fs::path& config_file_path);

        /**
         * @brief Computes the control action given the current state and contacts.
         *
         * @param target_state the desired state of the robot. A vector (q, v).
         * @param state the current state of the robot. A vector (q, v).
         * @param contact the contacts to enforce as holonomic constraints
         * @return a vector of motor torques
         */
        vectorx_t ComputeControl(const vectorx_t& target_state, const vectorx_t& force_target,
                                 const vectorx_t& state, const models::RobotContactInfo& contact);

    protected:
        /**
         * @brief parses the file given by config_file_path_ and assigns the gains/settings.
         */
        void ParseUpdateSettings();

        // ---------------------------------------- //
        // --------- Constraint Functions --------- //
        // ---------------------------------------- //
        void AddDynamicsConstraints(const vectorx_t& q, const vectorx_t& v, const models::RobotContactInfo& contact);
        void AddHolonomicConstraints(const vectorx_t& q, const vectorx_t& v, const models::RobotContactInfo& contact);
        void AddTorqueConstraints(const vectorx_t& q, const vectorx_t& v, const models::RobotContactInfo& contact);
        void AddFrictionConeConstraints(const models::RobotContactInfo& contact);
        void AddPositiveGRFConstraints(const models::RobotContactInfo& contact);

        // ---------------------------------------- //
        // ------------ Cost Functions ------------ //
        // ---------------------------------------- //
        void AddLegTrackingCost(const vectorx_t& q, const vectorx_t& v);
        void AddTorsoTrackingCost(const vectorx_t& q, const vectorx_t& v);
        void AddForceTrackingCost(const vectorx_t& q, const vectorx_t& v);

        // ---------------------------------------- //
        // ----------- Helper Functions ----------- //
        // ---------------------------------------- //
        [[nodiscard]] int DynamicsIdx() const;
        [[nodiscard]] int HolonomicIdx() const;
        [[nodiscard]] int TorqueIdx() const;
        [[nodiscard]] int FrictionIdx() const;
        [[nodiscard]] int PosGRFIdx() const;

        // -------- Constants ------- //
        static constexpr int FLOATING_VEL_OFFSET = 6;
        static constexpr int FLOATING_BASE_OFFSET = 7;
        static constexpr int POINT_CONTACT_CONSTRAINTS = 3;
        static constexpr int POS_VARS = 3;

        // -------- Member Variables -------- //
        fs::path config_file_path_;
        std::string name_;

        vectorx_t q_target_;
        vectorx_t v_target_;
        vectorx_t force_target_;

        // -------- QP information -------- //
        solvers::OSQPInterface qp_solver_;
        constraints::BoxConstraints constraints_;
        matrixx_t P_;
        vectorx_t w_;

        // -------- Constraint settings -------- //
        vectorx_t torque_bounds_;
        double friction_coef_;
        double max_grf_;

        // -------- Cost settings -------- //
        double leg_tracking_weight_;
        double torso_tracking_weight_;
        double force_tracking_weight_;
        vectorx_t kd_joint_;
        vectorx_t kp_joint_;
        double kp_pos_;
        double kp_orientation_;
        double kd_pos_;
        double kd_orientation_;

        // Dynamics terms
        matrixx_t M_;
        matrixx_t C_;
        vectorx_t g_;

        // Contact info
        int num_contacts_;

        std::unique_ptr<models::FullOrderRigidBody> model_;
    private:
        void ConvertStdVectorToEigen(const std::vector<double>& v1, vectorx_t& v2);
    };
}


#endif //TORC_WHOLE_BODY_QP_CONTROLLER_H
