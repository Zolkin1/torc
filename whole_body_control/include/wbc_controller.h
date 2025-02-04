//
// Created by zolkin on 2/3/25.
//

#ifndef WBC_CONTROLLER_H
#define WBC_CONTROLLER_H
#include "osqp++.h"

#include "cpp_ad_interface.h"
#include "full_order_rigid_body.h"

namespace torc::controller {
    using vectorx_t = Eigen::VectorXd;
    using vector4_t = Eigen::Vector4d;
    using matrixx_t = Eigen::MatrixXd;
    using matrix6_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;

    class WbcController {
    public:
        WbcController(const models::FullOrderRigidBody& model, const std::vector<std::string>& contact_frames,
            const vectorx_t& base_weight, const vectorx_t& joint_weight,
            const vectorx_t& tau_weight, const vectorx_t& force_weight,
            const vectorx_t& kp, const vectorx_t& kd,
            double friction_coef, bool verbose,
            const std::filesystem::path deriv_lib_path, bool compile_derivs);

        vectorx_t ComputeControl(const vectorx_t& q, const vectorx_t& v,
            const vectorx_t& q_des, const vectorx_t& v_des, const vectorx_t& tau_des, const vectorx_t& F_des,
            std::vector<bool> in_contact);

    protected:
        matrixx_t HolonomicConstraint(const vectorx_t& q, const vectorx_t& v,
            const std::vector<bool>& in_contact, vectorx_t& lb, vectorx_t& ub);

        matrixx_t ForceConstraint(const std::vector<bool>& in_contact, vectorx_t& lb, vectorx_t& ub);

        matrixx_t DynamicsConstraint(const vectorx_t& q, const vectorx_t& v, vectorx_t& lb, vectorx_t& ub);

        matrixx_t TorqueBoxConstraint() const;

        std::pair<matrixx_t, vectorx_t> StateTracking(const vectorx_t& q, const vectorx_t& v, const vectorx_t& q_des, const vectorx_t& v_des);

        std::pair<matrixx_t, vectorx_t> TorqueTracking(const vectorx_t& tau_des) const;

        std::pair<matrixx_t, vectorx_t> ForceTracking(const vectorx_t& F_des) const;

        void DynamicsFunction(const ad::ad_vector_t& a_tau_F, const ad::ad_vector_t& q_v, ad::ad_vector_t& violation);

        bool verbose_;

        vectorx_t kp_, kd_;
        vectorx_t base_weight_, joint_weight_, tau_weight_, force_weight_;
        double mu_;

        // Model
        std::vector<std::string> contact_frames_;
        models::FullOrderRigidBody model_;
        pinocchio::Data pin_data_;
        int nv_;
        int ntau_;
        int nF_;
        int ncontact_frames_;
        int ncontacts_;

        // OSQP Interface
        osqp::OsqpInstance osqp_instance_;
        osqp::OsqpSolver osqp_solver_;
        osqp::OsqpSettings osqp_settings_;

        int nd_;    // Number of decision variables
        int nc_;    // Number of constraints

        // Autodiff
        std::unique_ptr<ad::CppADInterface> inverse_dynamics_;

        static constexpr int FLOATING_VEL = 6;
        static constexpr int CONTACT_3DOF = 3;
    private:
    };
}


#endif //WBC_CONTROLLER_H
