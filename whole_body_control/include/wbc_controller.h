//
// Created by zolkin on 2/3/25.
//

#ifndef WBC_CONTROLLER_H
#define WBC_CONTROLLER_H

#include <fstream>

#include <proxsuite/proxqp/dense/dense.hpp>

#include "cpp_ad_interface.h"
#include "full_order_rigid_body.h"

namespace torc::controller {
    using vectorx_t = Eigen::VectorXd;
    using vector3_t = Eigen::Vector3d;
    using vector4_t = Eigen::Vector4d;
    using matrixx_t = Eigen::MatrixXd;
    using matrix6_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;

    class WbcSettings {
        public:
        WbcSettings(const std::filesystem::path& config_file);

        vectorx_t base_weight, joint_weight, tau_weight, force_weight;
        vectorx_t kp, kd;
        bool verbose;
        std::vector<std::string> tracking_frames_;
        std::vector<vector3_t> tracking_weights_;
        std::vector<vector3_t> tracking_kp;
        std::vector<vector3_t> tracking_kd;

        std::vector<std::string> skip_joints;
        std::vector<double> joint_values;
        std::vector<std::string> contact_frames;

        vectorx_t custom_torque_lims;

        bool log;
        int log_period;

        bool compile_derivs;
        double alpha;
    };

    class WbcController {
    public:
        WbcController(const models::FullOrderRigidBody& model, const std::vector<std::string>& contact_frames,
            WbcSettings settings,
            double friction_coef, bool verbose,
            const std::filesystem::path deriv_lib_path, bool compile_derivs);

        ~WbcController();

        vectorx_t ComputeControl(const vectorx_t& q, const vectorx_t& v,
            const vectorx_t& q_des, const vectorx_t& v_des, const vectorx_t& tau_des, const vectorx_t& F_des,
            std::vector<bool> in_contact);

    protected:
        matrixx_t HolonomicConstraint(const vectorx_t& q, const vectorx_t& v,
            const std::vector<bool>& in_contact, vectorx_t& b);

        matrixx_t ForceConstraint(const std::vector<bool>& in_contact, vectorx_t& lb, vectorx_t& ub);

        matrixx_t NoForceConstraint(const std::vector<bool>& in_contact, vectorx_t& b);

        matrixx_t DynamicsConstraint(const vectorx_t& q, const vectorx_t& v, vectorx_t& b);

        matrixx_t TorqueBoxConstraint() const;

        std::pair<matrixx_t, vectorx_t> StateTracking(const vectorx_t& q, const vectorx_t& v, const vectorx_t& q_des, const vectorx_t& v_des);

        std::pair<matrixx_t, vectorx_t> TorqueTracking(const vectorx_t& tau_des);

        std::pair<matrixx_t, vectorx_t> ForceTracking(const vectorx_t& F_des);

        std::pair<matrixx_t, vectorx_t> FrameTracking(const vectorx_t& q, const vectorx_t& v,
            const vectorx_t& q_des, const vectorx_t& v_des, const std::vector<bool>& in_contact);

        void FrameTrackingFunction(const std::string& frame,
            const ad::ad_vector_t& a, const ad::ad_vector_t& q_v_kp_kd, ad::ad_vector_t& violation);

        void DynamicsFunction(const ad::ad_vector_t& a_tau_F, const ad::ad_vector_t& q_v, ad::ad_vector_t& violation);

        void LogEigenVec(const vectorx_t& vec);

        bool verbose_;

        WbcSettings settings_;

        // vectorx_t kp_, kd_;
        // vectorx_t base_weight_, joint_weight_, tau_weight_, force_weight_;
        double mu_;

        // Model
        std::vector<std::string> contact_frames_;
        models::FullOrderRigidBody model_;
        pinocchio::Data pin_data_;
        int nq_;
        int nv_;
        int ntau_;
        int nF_;
        int ncontact_frames_;
        int ncontacts_;

        int nd_;    // Number of decision variables
        int nc_;    // Number of constraints

        // Autodiff
        std::unique_ptr<ad::CppADInterface> inverse_dynamics_;
        // std::map<std::string, std::unique_ptr<ad::CppADInterface>> frame_tracking_;

        static constexpr int FLOATING_VEL = 6;
        static constexpr int CONTACT_3DOF = 3;

        // Logging
        unsigned long solve_count_;
        std::ofstream log_file_;
    private:
    };
}


#endif //WBC_CONTROLLER_H
