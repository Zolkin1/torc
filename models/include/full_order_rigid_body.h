//
// Created by zolkin on 5/23/24.
//

#ifndef TORC_FULL_ORDER_RIGID_BODY_H
#define TORC_FULL_ORDER_RIGID_BODY_H

#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"

#include "pinocchio_model.h"
#include "robot_contact_info.h"

namespace torc::models {
    using vectorx_t = Eigen::VectorXd;
    using matrixx_t = Eigen::MatrixXd;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    using matrix3x_t = Eigen::Matrix<double, 3, Eigen::Dynamic>;

    template <typename ScalarT>
    struct ExternalForce {
        std::string frame_name;
        Eigen::Vector3<ScalarT> force_linear;

        ExternalForce(const std::string& frame, const Eigen::Vector3<ScalarT>& force) {
            frame_name = frame;
            force_linear = force;
        }
    };

    class FullOrderRigidBody : public PinocchioModel {
    public:

        FullOrderRigidBody(const std::string& name, const std::filesystem::path& model_path, bool urdf_model=true);

        FullOrderRigidBody(const std::string& name, const std::filesystem::path& model_path,
                           const std::vector<std::string>& underactuated_joints, bool urdf_model=true);

        FullOrderRigidBody(const std::string& name, const std::filesystem::path& model_path,
            const std::vector<std::string>& joint_skip_names, const std::vector<double>& joint_skip_values);

        FullOrderRigidBody(const std::string& name, const std::filesystem::path& model_path,
            const std::vector<std::string>& joint_skip_names, const std::vector<double>& joint_skip_values,
            std::vector<std::string> new_frame_names, std::vector<vector3_t> new_frame_positions);

        FullOrderRigidBody(const FullOrderRigidBody& other);

        [[nodiscard]] int GetStateDim() const override;

        [[nodiscard]] int GetDerivativeDim() const override;

        [[nodiscard]] vectorx_t GetRandomState() const override;

        [[nodiscard]] quat_t GetBaseOrientation(const vectorx_t& q) const override;

        [[nodiscard]] vectorx_t IntegrateVelocity(const vectorx_t& q0, const vectorx_t& v) const;

        // @note These are not actually const functions as we modify the pin_data struct
        [[nodiscard]] vectorx_t GetDynamics(const vectorx_t& state,
                                            const vectorx_t& input) override;

        [[nodiscard]] vectorx_t GetDynamics(const vectorx_t& q, const vectorx_t& v,
                                            const vectorx_t& input,
                                            const std::vector<ExternalForce<double>>& f_ext);

        [[nodiscard]] vectorx_t GetDynamics(const vectorx_t& state,
                                            const vectorx_t& input,
                                            const RobotContactInfo& contact_info) const;

        [[nodiscard]] vectorx_t InverseDynamics(const vectorx_t& q, const vectorx_t& v, const vectorx_t& a,
                                                const std::vector<ExternalForce<double>>& f_ext);
//                                                const pinocchio::container::aligned_vector<pinocchio::Force>& forces);

        vectorx_t GetImpulseDynamics(const vectorx_t& state,
                                     const vectorx_t& input,
                                     const RobotContactInfo& contact_info);

        // TODO: Linearization function that calculates the FD and derivatives so we don't have redundant calls

        void DynamicsDerivative(const vectorx_t& state,
                                const vectorx_t& input,
                                matrixx_t& A,
                                matrixx_t& B) override;


        void DynamicsDerivative(const vectorx_t& state,
                                const vectorx_t& input,
                                const RobotContactInfo& contacts,
                                matrixx_t& A,
                                matrixx_t& B);

        void InverseDynamicsDerivative(const vectorx_t& q,
                                       const vectorx_t& v,
                                       const vectorx_t& a,
//                                       const pinocchio::container::aligned_vector<pinocchio::Force>& forces,
                                       const std::vector<ExternalForce<double>>& f_ext,
                                       matrixx_t& dtau_dq,
                                       matrixx_t& dtau_dv,
                                       matrixx_t& dtau_da,
                                       matrixx_t& dtau_df);

        void ImpulseDerivative(const vectorx_t& state,
                               const vectorx_t& input,
                               const RobotContactInfo& contact_info,
                               matrixx_t& A,
                               matrixx_t& B);

        void GetDynamicsTerms(const vectorx_t& state, matrixx_t& M, matrixx_t& C, vectorx_t& g);

        pinocchio::Motion GetFrameAcceleration(const std::string& frame);

        void ParseState(const vectorx_t& state,
                        vectorx_t& q,
                        vectorx_t& v) const;

        void ParseStateDerivative(const vectorx_t& dstate,
                                  vectorx_t& v,
                                  vectorx_t& a) const;

        static vectorx_t BuildState(const vectorx_t& q, const vectorx_t& v);

        static vectorx_t BuildStateDerivative(const vectorx_t& v, const vectorx_t& a);

        void ParseInput(const vectorx_t& input, vectorx_t& tau) const;

        vector3_t QuaternionIntegrationRelative(const quat_t& qbar_kp1, const quat_t& qbar_k, const vector3_t& xi,
            const vector3_t& w, double dt);

        void FrameVelDerivWrtConfiguration(const vectorx_t& q,
            const vectorx_t& v, const vectorx_t& a, const std::string& frame, matrix6x_t& jacobian,
            const pinocchio::ReferenceFrame& ref = pinocchio::LOCAL_WORLD_ALIGNED);

        void PerturbConfiguration(vectorx_t& q, double delta, int idx);

        /**
         * Takes the torques on the actuated coordinates and maps to a vector of
         * dimension model.nv with zeros on underacutated joints
         * @param input
         * @return full input vector
         */
        [[nodiscard]] vectorx_t InputsToTau(const vectorx_t& input) const override;

//        [[nodiscard]] vectorx_t TauToInputs(const vectorx_t& tau) const;

        [[nodiscard]] pinocchio::container::aligned_vector<pinocchio::Force> ConvertExternalForcesToPin(const vectorx_t& q,
                                                                                                        const std::vector<ExternalForce<double>>& f_ext) const;

        matrixx_t ExternalForcesDerivativeWrtConfiguration(const vectorx_t& q, const std::vector<ExternalForce<double>>& f_ext);

        /**
        * Compute a configuration that respects the task space constraints.
        * @param positions 3d positions for certain frames
        * @param frames corresponding frame names
        */
        vectorx_t InverseKinematics(const vectorx_t& base_config, const std::vector<vector3_t>& positions, const std::vector<std::string>& frames,
            const vectorx_t& q_guess, bool use_floating_base=false);

        quat_t PoseFit(const vector3_t& base_position, const quat_t& quat_guess, const std::vector<vector3_t>& frame_positions, const std::vector<std::string>& frames);

        // DEBUG ------
        pinocchio::Model GetModel() const;
        // ------------

        constexpr static size_t STATE_Q_IDX = 0;
        constexpr static size_t STATE_V_IDX = 1;

    protected:
        void CreateActuationMatrix(const std::vector<std::string>& underactuated_joints);

        matrixx_t act_mat_;

        std::unique_ptr<pinocchio::Data> contact_data_;
    private:

    };
} // torc::models


#endif //TORC_FULL_ORDER_RIGID_BODY_H
