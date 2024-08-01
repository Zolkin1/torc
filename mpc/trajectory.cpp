//
// Created by zolkin on 6/20/24.
//

#include "trajectory.h"

namespace torc::mpc {
    void Trajectory::UpdateSizes(int config_size, int vel_size, int tau_size, const std::vector<std::string>& force_frames, int nodes) {
        config_size_ = config_size;
        vel_size_ = vel_size;
        tau_size_ = tau_size;
        int idx = 0;
        for (const auto& frame : force_frames) {
            force_frames_.insert(std::pair<std::string, int>(frame, idx));
            idx++;
        }

        num_frames_ = force_frames.size();

        SetNumNodes(nodes);

        for (int node = 0; node < nodes_; node++) {
            q_[node].setZero(config_size_);
            v_[node].setZero(vel_size_);
            tau_[node].setZero(tau_size_);
            for (int frame = 0; frame < num_frames_; frame++) {
                forces_[node][frame].setZero();
            }
        }
    }

    int Trajectory::GetNumNodes() const {
        return nodes_;
    }

    void Trajectory::SetNumNodes(int nodes) {
        nodes_ = nodes;
        q_.resize(nodes_);
        v_.resize(nodes_);
        tau_.resize(nodes_);
        forces_.resize(nodes_);
        for (auto& force : forces_) {
            force.resize(num_frames_);
        }
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

    void Trajectory::SetForce(int node, const std::string& frame, const torc::mpc::vector3_t& f) {
        forces_[node][force_frames_[frame]] = f;
    }

    vectorx_t Trajectory::GetConfiguration(int node) {
        return q_[node];
    }

    quat_t Trajectory::GetQuat(int node) {
        return static_cast<quat_t>(q_[node].segment<4>(3));
    }


    vectorx_t Trajectory::GetVelocity(int node) {
        return v_[node];
    }

    vectorx_t Trajectory::GetTau(int node) {
        return tau_[node];
    }

    vector3_t Trajectory::GetForce(int node, const std::string& frame) {
        return forces_[node][force_frames_[frame]];
    }

    void Trajectory::SetDefault(const vectorx_t& q_default) {
        for (int node = 0; node < nodes_; node++) {
            q_[node] = q_default;
            v_[node].setZero();
            tau_[node].setZero();
            for (int frame = 0; frame < num_frames_; frame++) {
                forces_[node][frame].setZero();
            }
        }
    }

} //torc::mpc