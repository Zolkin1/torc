//
// Created by zolkin on 6/18/24.
//

#ifndef TORC_MPC_BASE_H
#define TORC_MPC_BASE_H

#include "base_model.h"
#include "trajectory.h"

namespace torc::mpc {
    // TODO: remove

    /*
     * Take in the model, other constraints, and the cost
     *
     * Take in the current trajectory. Linearize/quadraticize about this trajectory
     *  - If there are impulse dynamics then I can choose to insert a new node,
     *      or change one of the other nodes to be the impulse dynamics
     *  - If contacts play a role, then I need a contact trajectory and I need to call correct functions
     * Create QP mats, convert trajectory into QP warm start
     * Solve the QP
     * Convert QP solution into new trajectory. Line search if desired.
     * Return new trajectory
     * TODO: Need to figure out how I want to deal with different model requirements
     * TODO: Figure out how I want to support users interfacing with their own solvers.
     *  I think this can be solved by the user passing a function pointer to their own solver
     *  interface. They will also need to specify a data struct type (one of the common ones).
     *
     *  MPC will provide functions that return (via an argument) each of the different QP data types.
     *  The user will Call this function and the MPC will provide all the data formatted to go into the QP
     *  Then the user will need to call the QP solve (MPC can also provide a function to get a QP warm start).
     *  Then the user will pass the QP result back to MPC and we can process it into a trajectory and perform
     *  a line search. This will return a new trajectory.
     *
     *  We will provide a call to perform the IPOPT solve if the user does not want to use RTI.
     *
     *  MPC will always structure the QP vector as [states, inputs]^T (when it can).
     *
     *  TODO: For now I will just write an MPC implementation that uses one model and interfaces with one
     *   solver. Once that is done I will be able to see more clearly what is needed.
     *   One option is making base model have some of the functions I need (abstract), then the contact
     *   and smooth interfaces and inherit from base model. PinocchioModel would not inherit from BaseModel.
     *   User models would need to inherit from the interfaces that inherit from BaseModel.
     */
    class MPCBase {
    public:
//        MPCBase();

// Note that we can always ignore the contact part of the trajectory because we keep that constant.
// So, these functions can return the state and input part of the trajectory only.
        /**
         * Converts the result of the QP back into trajectory form
         * @param qp_vec
         * @param traj
         */
        void ConvertQPVecToTraj(const vectorx_t& qp_vec, Trajectory& traj) const;

        void ConvertQPVecToTraj(const vectorx_t& qp_vec, std::vector<models::RobotState>& states,
            std::vector<vectorx_t>& inputs) const;

        /**
         * Line search on the QP result then convert back to the trajectory
         * @param qp_vec
         * @param traj
         */
        void LineSearch(const vectorx_t& qp_vec, Trajectory& traj) const;

        void LineSearch(const vectorx_t& qp_vec, std::vector<models::RobotState>& states,
                        std::vector<vectorx_t>& inputs) const;

    protected:
        // Hold a constraint object

        // Hold a cost object

        // Hold all the metadata associated with the constraints and dynamics

        int nodes_;

    private:
    };
} // torc::mpc


#endif //TORC_MPC_BASE_H
