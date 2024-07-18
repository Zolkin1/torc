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
     * a seperated FullOrderRigidBody object needs to be made.
     *
     * The inputs are:
     * [f0, ... , fl, r0, ... , rl]
     * Where l = number of contacts. Each fi and ri \in R^3.
     *
     * The tau on the SRB are the forces and moments.
     * Forces are trivial and moments can be computed via cross products.
     */
    class SingleRigidBody : public PinocchioModel {

    public:
        SingleRigidBody(const std::string& name,
                        const std::filesystem::path& urdf,
                        int max_contacts);

        SingleRigidBody(const std::string& name,
                        const std::filesystem::path& urdf,
                        const vectorx_t& ref_config,
                        int max_contacts);

        [[nodiscard]] int GetStateDim() const override;

        [[nodiscard]] int GetDerivativeDim() const override;

        [[nodiscard]] vectorx_t GetRandomState() const override;

        [[nodiscard]] quat_t GetBaseOrientation(const vectorx_t& q) const override;

        void SetRefConfig(const vectorx_t& ref_config);

        vectorx_t GetDynamics(const vectorx_t& state,
                              const vectorx_t& input) override;

        void DynamicsDerivative(const vectorx_t& state,
                                const vectorx_t& input,
                                matrixx_t& A,
                                matrixx_t& B) override;

        static void ParseState(const vectorx_t& state, vectorx_t& q, vectorx_t& v);

        static void ParseStateDerivative(const vectorx_t& dstate, vectorx_t& v, vectorx_t& a);

        [[nodiscard]] vectorx_t GetRefConfig() const;

        static constexpr int SRB_CONFIG_DIM = 7;
        static constexpr int SRB_VEL_DIM = 6;

    protected:

        [[nodiscard]] vectorx_t InputsToTau(const vectorx_t& input) const override;

        [[nodiscard]] matrixx_t ActuationMapDerivative(const vectorx_t& input,
                                         bool force_and_pos = true) const;

        pinocchio::Model full_pin_model_;
        std::unique_ptr<pinocchio::Data> full_pin_data_;

        vectorx_t ref_config_;
        int max_contacts_;

    private:
        void MakeSingleRigidBody(const vectorx_t& ref_config,
                                 bool reassign_full_model = true);
    };
} // torc::models


#endif //TORC_SINGLE_RIGID_BODY_H
