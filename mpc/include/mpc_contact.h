//
// Created by zolkin on 6/18/24.
//

#ifndef TORC_MPC_CONTACT_H
#define TORC_MPC_CONTACT_H

#include <memory>

#include "contact_model.h"
#include "rigid_body.h"

#include "trajectory.h"

namespace torc::mpc {
    class MPCContact {
    public:
        using matrtixx_t = Eigen::MatrixXd;
        using vectorx_t = Eigen::VectorXd;

        // TODO: Add costs and constraints
        MPCContact(const models::RigidBody& model, int nodes, bool insert_nodes = false);

        // Want to provide this function for all the generic data types and any of the special ones
        //  that I provide an interface for.
        void ToHPIPMData(const Trajectory& traj);

        void ToBilateralData(const ContactTrajectory& traj);

    protected:
        [[nodiscard]] bool VerifyTrajectory(const ContactTrajectory& traj) const;

        void GetLinearization(const ContactTrajectory& traj, int node,
                              matrtixx_t& A, matrtixx_t&  B, vectorx_t& c);

        models::RigidBody model_;

        // Defines how many state nodes there are. Should be nodes_ - 1 inputs
        int nodes_;

        bool insert_nodes_for_impulse_;

        // cost fcn

        // constraints

        // sparse matrix builder

    private:
    };
} // torc::mpc


#endif //TORC_MPC_CONTACT_H
