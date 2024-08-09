//
// Created by zolkin on 6/20/24.
//

#include <iostream>
#include "pinocchio/math/quaternion.hpp"
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
        dt_.resize(nodes_);
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

    void Trajectory::SetDtVector(const std::vector<double>& dt) {
        if (dt.size() != q_.size() - 1) {
            std::cerr << "dt vector does not have the correct number of nodes! Ignoring!" << std::endl;
        }
        dt_ = dt;
    }


    vectorx_t Trajectory::GetConfiguration(int node) const {
        return q_[node];
    }

    quat_t Trajectory::GetQuat(int node) const {
        return static_cast<quat_t>(q_[node].segment<4>(3));
    }


    vectorx_t Trajectory::GetVelocity(int node) const {
        return v_[node];
    }

    vectorx_t Trajectory::GetTau(int node) const {
        return tau_[node];
    }

    vector3_t Trajectory::GetForce(int node, const std::string& frame) const {
        return forces_[node][force_frames_.at(frame)];
    }

    const std::vector<double> &Trajectory::GetDtVec() const {
        return dt_;
    }


    // -------------------------- //
    // ----- Interpolations ----- //
    // -------------------------- //
    void Trajectory::GetConfigInterp(double time, vectorx_t& q_out) {
        if (time < 0) {
            std::cerr << "Interpolation time < 0! Returning 0!" << std::endl;
            q_out = vectorx_t::Zero(q_[0].size());
            return;
        }

        q_out.resize(q_[0].size());

        double time_tally = dt_[0];
        for (int i = 1; i < dt_.size(); i++) {
            if (time < time_tally) {
                // We know the nodes that it is between
                double forward_weight = 1.0 - (time_tally - time)/dt_[i-1];
                // std::cout << "forward weight: " << forward_weight << std::endl;
                // std::cout << "time tally: " << time_tally << std::endl;
                // std::cout << "time: " << time << std::endl;
                // std::cout << "dt: " << dt_[i-1] << std::endl;

                // Some weird rounding errors sometimes occur
                if (forward_weight > 1 && forward_weight < 1 + 1e-10) {
                    forward_weight = 1;
                }

                if (forward_weight < 0 && forward_weight > 0 - 1e-10) {
                    forward_weight = 0;
                }

                if (forward_weight > 1 || forward_weight < 0) {
                    throw std::runtime_error("[Trajectory] Interpolation invalid times!");
                }

                q_out.head<POS_VARS>() = forward_weight*q_[i].head<POS_VARS>() - (1 - forward_weight)*q_[i-1].head<POS_VARS>();
                const size_t num_joints = q_[0].size() - FLOATING_BASE;
                q_out.tail(num_joints) = forward_weight*q_[i].tail(num_joints) - (1 - forward_weight)*q_[i-1].tail(num_joints);

                // Quaternion
                quat_t quat_out;
                pinocchio::quaternion::slerp(forward_weight, GetQuat(i-1), GetQuat(i), quat_out);
                q_out.segment<QUAT_VARS>(POS_VARS) = quat_out.coeffs();
                return;
            }

            time_tally += dt_[i];
        }

        std::cerr << "Time is too large! No interpolation provided!" << std::endl;
    }

    void Trajectory::GetVelocityInterp(double time, vectorx_t& v_out) {
        StandardVectorInterp(time, v_out, v_);
    }

    void Trajectory::GetTorqueInterp(double time, vectorx_t& torque_out) {
        StandardVectorInterp(time, torque_out, tau_);
    }

    void Trajectory::GetForceInterp(double time, const std::string& frame, vector3_t& force_out) {
        if (time < 0) {
            std::cerr << "Interpolation time < 0! No interpolation provided!" << std::endl;
            return;
        }

        double time_tally = dt_[0];
        for (int i = 1; i < dt_.size(); i++) {
            if (time < time_tally) {
                // We know the nodes that it is between
                double forward_weight = 1.0 - (time_tally - time)/dt_[i-1];

                // Some weird rounding errors sometimes occur
                if (forward_weight > 1 && forward_weight < 1 + 1e-10) {
                    forward_weight = 1;
                }

                if (forward_weight < 0 && forward_weight > 0 - 1e-10) {
                    forward_weight = 0;
                }

                if (forward_weight > 1 || forward_weight < 0) {
                    throw std::runtime_error("[Trajectory] Interpolation invalid times!");
                }

                force_out = forward_weight*forces_[i][force_frames_[frame]] - (1 - forward_weight)*forces_[i-1][force_frames_[frame]];
                return;
            }

            time_tally += dt_[i];
        }

        std::cerr << "Interpolation time is too large! No interpolation provided!" << std::endl;
    }


    void Trajectory::StandardVectorInterp(double time, vectorx_t& vec_out, const std::vector<vectorx_t>& vecs) {
        if (time < 0) {
            std::cerr << "Interpolation time < 0! No interpolation provided!" << std::endl;
            return;
        }

        double time_tally = dt_[0];
        for (int i = 1; i < dt_.size(); i++) {
            if (time < time_tally) {
                // We know the nodes that it is between
                double forward_weight = 1.0 - (time_tally - time)/dt_[i-1];

                // Some weird rounding errors sometimes occur
                if (forward_weight > 1 && forward_weight < 1 + 1e-10) {
                    forward_weight = 1;
                }

                if (forward_weight < 0 && forward_weight > 0 - 1e-10) {
                    forward_weight = 0;
                }

                if (forward_weight > 1 || forward_weight < 0) {
                    throw std::runtime_error("[Trajectory] Interpolation invalid times!");
                }

                vec_out = forward_weight*vecs[i] - (1 - forward_weight)*vecs[i-1];
                return;
            }

            time_tally += dt_[i];
        }

        std::cerr << "Interpolation time is too large! No interpolation provided!" << std::endl;
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