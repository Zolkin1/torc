//
// Created by zolkin on 6/20/24.
//

#include "trajectory.h"

namespace torc::mpc {
    int Trajectory::GetNumNodes() const {
        return nodes_;
    }

    void Trajectory::SetNumNodes(int nodes) {
        nodes_ = nodes;
        q_.resize(nodes_);
        v_.resize(nodes_);
        tau_.resize(nodes_);
        forces_.resize(nodes_);
    }

    void Trajectory::SetConfiguration(int node, const vectorx_t& q) {
        q_[node] = q;
    }

    void Trajectory::SetVelocity(int node, const torc::mpc::vectorx_t& v) {
        v_[node] = v;
    }

    void Trajectory::SetTau(int node, const torc::mpc::vectorx_t& tau) {
        tau_[node] = tau;
    }

    void Trajectory::SetForce(int node, const torc::mpc::vectorx_t& f) {
        forces_[node] = f;
    }


} //torc::mpc