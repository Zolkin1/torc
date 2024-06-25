//
// Created by zolkin on 6/18/24.
//

#include "mpc_contact.h"

namespace torc::mpc {
    MPCContact::MPCContact(const models::RigidBody& model, int nodes, bool insert_nodes)
        : model_(model), nodes_(nodes), insert_nodes_for_impulse_(insert_nodes) {
    }

    void MPCContact::ToHPIPMData(const torc::mpc::Trajectory& traj) {

    }

    void MPCContact::ToBilateralData(const torc::mpc::ContactTrajectory& traj) {
        if (!VerifyTrajectory(traj)) {
            throw std::runtime_error("Invalid trajectory.");
        }

        const int deriv_dim = model_.GetDerivativeDim();

        matrtixx_t A = matrtixx_t::Zero(deriv_dim, deriv_dim);
        matrtixx_t B = matrtixx_t::Zero(deriv_dim, model_.GetNumInputs());
        vectorx_t c = vectorx_t::Zero(deriv_dim);

        std::vector<int> replacement_nodes;
        if (!insert_nodes_for_impulse_ && !traj.impulse_times.empty()) {
            // Determine which nodes will be switched
            for (int impulse = 0; impulse < traj.impulse_times.size(); impulse++) {
                double min_diff = 1e8;
                int replace_node = 0;
                for (int node = 0; node < nodes_; node++) {
                    if (std::abs((traj.dt * node) - traj.impulse_times.at(impulse)) < min_diff) {
                        min_diff = std::abs((traj.dt * node) - traj.impulse_times.at(impulse));
                        replace_node = node;
                    }
                }

                replacement_nodes.emplace_back(replace_node);
            }
        }

        // TODO: Provide multithreading support
        for (int node = 0; node < nodes_; node++) {
            // Get the linearization of the dynamics
            if (insert_nodes_for_impulse_) {
                // Insert nodes into the trajectory
                throw std::runtime_error("Node insertion not supported yet.");
            } else if (!replacement_nodes.empty()) {
                // There are impulses in the trajectory, see if the current node is replaced
                const auto it = std::find(replacement_nodes.begin(),
                                          replacement_nodes.end(), node);
                if (it == replacement_nodes.end()) {
                    // Current node is not replaced, linearize as normal
                    GetLinearization(traj, node, A, B, c);
                } else {
                    // Current node is replaced, use impulse dynamics
                    // TODO: Do I need to mark the timing here so that the output trajectory
                    //  can be re-constructed properly?
                    model_.ImpulseDerivative(traj.states.at(node), traj.inputs.at(node),
                                             traj.contacts.at(node), A, B);

                    models::RobotState state = model_.GetImpulseDynamics(traj.states.at(node),
                                                                         traj.inputs.at(node),
                                                                         traj.contacts.at(node));

                    // TODO: Need to convert this to the lie algebra otherwise it will be the wrong dimension
                    state.ToVector(c);
                }
            } else {
                GetLinearization(traj, node, A, B, c);
            }

            // Get linearization of the constraints

            // Get quadratic approx of the cost function

            // Insert into the bilateral data structure
        }

    }

    bool MPCContact::VerifyTrajectory(const torc::mpc::ContactTrajectory& traj) const {
        if (traj.states.size() != nodes_) {
            return false;
        }

        if (traj.inputs.size() != nodes_ - 1) {
            return false;
        }

        if (traj.contacts.size() != nodes_ - 1) {
            return false;
        }

        // TODO: Can consider checking this for each node of the trajectory
        if (traj.states.at(0).q.size() + traj.states.at(0).v.size() != model_.GetStateDim()) {
            return false;
        }

        // TODO: Can consider checking this for each node of the trajectory
        if (traj.inputs.at(0).size() != model_.GetNumInputs()) {
            return false;
        }

        return true;
    }

    void MPCContact::GetLinearization(const torc::mpc::ContactTrajectory& traj, int node,
                                      torc::mpc::MPCContact::matrtixx_t& A,
                                      torc::mpc::MPCContact::matrtixx_t& B,
                                      torc::mpc::MPCContact::vectorx_t& c) {
        models::RobotStateDerivative deriv = model_.GetDynamics(traj.states.at(node),
                                                                traj.inputs.at(node),
                                                                traj.contacts.at(node));

        deriv.ToVector(c);

        model_.DynamicsDerivative(traj.states.at(node),
                                  traj.inputs.at(node), traj.contacts.at(node),
                                  A, B);
    }
} // torc::mpc