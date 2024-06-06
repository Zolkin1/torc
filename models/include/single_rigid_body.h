//
// Created by zolkin on 6/4/24.
//

#ifndef TORC_SINGLE_RIGID_BODY_H
#define TORC_SINGLE_RIGID_BODY_H

#include "pinocchio_model.h"

namespace torc::models {

    struct SRBInput {
        vectorx_t forces_positions;
        int num_contacts;

        // Can provide helper functions to parse the vectors
    };

    /**
     * SingleRigidBody holds all the information for create a SRB model.
     * The SRB model is created by fixing all the joints in a reference configuration to get the single body.
     * Inputs are external forces applied at locations outside of the body.
     * SingleRigidBody does not expose any functions accessing the full rigid body dynamics. To access those,
     * a seperated RigidBody object needs to be made.
     *
     * The inputs are:
     * [f0, ... , fl, r0, ... , rl]
     * Where l = number of contacts. Each fi and ri \in R^3.
     *
     * The tau on the SRB are the forces and moments.
     * Forces are trivial and moments can be computed via cross products.
     */
    class SingleRigidBody : public PinocchioModel {
        using srb_config_t = Eigen::Vector<double, 7>;
        using srb_vel_t = Eigen::Vector<double, 6>;

    public:
        SingleRigidBody(const std::string& name, const std::filesystem::path& urdf);

        SingleRigidBody(const std::string& name, const std::filesystem::path& urdf,
                        const vectorx_t& ref_config);

        void SetRefConfig(const vectorx_t& ref_config);

        [[nodiscard]] vectorx_t GetRefConfig() const;

        [[nodiscard]] RobotStateDerivative GetDynamics(const RobotState& state, const vectorx_t& input) override;

        void DynamicsDerivative(const RobotState& state, const vectorx_t& input,
                                matrixx_t& A, matrixx_t& B) override;

        static constexpr int SRB_CONFIG_DIM = 7;
        static constexpr int SRB_VEL_DIM = 6;
    protected:

        [[nodiscard]] vectorx_t InputsToTau(const vectorx_t& input) const override;

        matrixx_t ActuationMapDerivative(const vectorx_t& input, bool force_and_pos = true) const;

        pinocchio::Model full_pin_model_;
        std::unique_ptr<pinocchio::Data> full_pin_data_;



        vectorx_t ref_config_;

    private:
        void MakeSingleRigidBody(const torc::models::PinocchioModel::vectorx_t& ref_config,
                                 bool reassign_full_model = true);
    };
} // torc::models


#endif //TORC_SINGLE_RIGID_BODY_H
