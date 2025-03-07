//
// Created by zolkin on 6/20/24.
//

#include <iostream>
#include "pinocchio/math/quaternion.hpp"
#include "trajectory.h"

#include <fstream>

namespace torc::mpc {
    Trajectory::Trajectory()
        : q_(0, 0), v_(0, 0), tau_(0, 0) {}

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

        q_.SetSizes(config_size_, nodes_);
        v_.SetSizes(vel_size_, nodes_);
        tau_.SetSizes(tau_size_, nodes_);

        for (int node = 0; node < nodes_; node++) {
            q_.InsertData(node, vectorx_t::Zero(config_size_));
            v_.InsertData(node, vectorx_t::Zero(vel_size_));
            tau_.InsertData(node, vectorx_t::Zero(tau_size_));
            for (int frame = 0; frame < num_frames_; frame++) {
                forces_[node][frame].setZero();
                in_contact_[node][frame] = true;   // Default to in contact
            }
        }
    }

    int Trajectory::GetNumNodes() const {
        return nodes_;
    }

    void Trajectory::SetNumNodes(int nodes) {
        nodes_ = nodes;
        q_.SetSizes(config_size_, nodes_);
        v_.SetSizes(vel_size_, nodes_);
        tau_.SetSizes(tau_size_, nodes_);
        forces_.resize(nodes_);
        in_contact_.resize(nodes_);
        dt_.resize(nodes_);
        for (auto& force : forces_) {
            force.resize(num_frames_);
        }

        for (auto& contact : in_contact_) {
            contact.resize(num_frames_);
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

    void Trajectory::SetInContact(int node, const std::string &frame, bool in_contact) {
        in_contact_[node][force_frames_[frame]] = in_contact;
    }

    void Trajectory::SetDtVector(const std::vector<double>& dt) {
        if (dt.size() != q_.GetNumNodes()) {
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

    std::vector<std::string> Trajectory::GetContactFrames() const {
        std::vector<std::string> frames;
        for (const auto& [frame, idx] : force_frames_) {
            frames.push_back(frame);
        }

        return frames;
    }

    double Trajectory::GetTotalTime() const {
        double total_time = 0;
        for (const auto& dt : dt_) {
            total_time += dt;
        }

        return  total_time;
    }

    bool Trajectory::GetInContact(const std::string &frame, int node) const {
        return in_contact_[node][force_frames_.at(frame)];
    }


    // -------------------------- //
    // ----- Interpolations ----- //
    // -------------------------- //
    bool Trajectory::GetInContactInterp(double time, const std::string &frame) {
        const auto node = GetNode(time);
        if (node.has_value()) {
            return in_contact_[node.value()][force_frames_[frame]];
        } else {
            throw std::runtime_error("Trajectory::GetInContact : invalid time!");
        }
    }


    void Trajectory::GetConfigInterp(double time, vectorx_t& q_out) {
        if (time < 0) {
            std::cerr << "Interpolation time < 0! Returning 0!" << std::endl;
            q_out = vectorx_t::Zero(q_[0].size());
            q_out(6) = 1;   // For a normalized quat
            return;
        }

        double time_tally = dt_[0];
        for (int i = 1; i < dt_.size(); i++) {
            if (time < time_tally) {
                q_out.resize(q_[0].size());

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

                q_out.head<POS_VARS>() = forward_weight*q_[i].head<POS_VARS>() + (1 - forward_weight)*q_[i-1].head<POS_VARS>();
                const size_t num_joints = q_[0].size() - FLOATING_BASE;
                q_out.tail(num_joints) = forward_weight*q_[i].tail(num_joints) + (1 - forward_weight)*q_[i-1].tail(num_joints);

                // Quaternion
                quat_t quat_out; // = GetQuat(i-1).slerp(forward_weight, GetQuat(i));
                pinocchio::quaternion::slerp(forward_weight, GetQuat(i-1), GetQuat(i), quat_out);
                q_out.segment<QUAT_VARS>(POS_VARS) = quat_out.coeffs();
                if (std::abs(q_out.segment<QUAT_VARS>(POS_VARS).norm() - 1) > 1e-2) {
                    std::cerr << "q: " << q_out.transpose() << std::endl;
                    throw std::runtime_error("[Trajectory] Interpolation invalid quaternion!");
                }
                return;
            }

            time_tally += dt_[i];
        }
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

                force_out = forward_weight*forces_[i][force_frames_[frame]] + (1 - forward_weight)*forces_[i-1][force_frames_[frame]];
                return;
            }

            time_tally += dt_[i];
        }
    }


    void Trajectory::StandardVectorInterp(double time, vectorx_t& vec_out, const SimpleTrajectory& vecs) {
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

                vec_out = forward_weight*vecs[i] + (1 - forward_weight)*vecs[i-1];
                return;
            }

            time_tally += dt_[i];
        }
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

    std::optional<int> Trajectory::GetNode(double time) const {
        double time_temp = 0;
        for (int node = 0; node < nodes_; node++) {
            if (time < time_temp) {
                return node;
            }
            time_temp += dt_[node];
        }

        return std::nullopt;
    }

    void Trajectory::ExportToCSV(const std::string &file_path) {
        std::cout << "Writing trajectory to " << file_path << std::endl;
        std::ofstream csv_file(file_path, std::ios_base::out);
        for (int node = 0; node < nodes_; node++) {
            vectorx_t q = q_[node];
            for (int i = 0; i < q.size(); i++) {
                csv_file << q[i] << ",";
            }

            vectorx_t v = v_[node];
            for (int i = 0; i < v.size(); i++) {
                csv_file << v[i] << ",";
            }

            vectorx_t tau = tau_[node];
            for (int i = 0; i < tau.size(); i++) {
                csv_file << tau[i] << ",";
            }
            for (int frame = 0; frame < num_frames_; frame++) {
                for (int i = 0; i < 3; i++) {
                    csv_file << forces_[node][frame][i] << ",";
                }
            }

            csv_file << dt_[node] << std::endl;
        }
        csv_file.close();
    }

    SimpleTrajectory Trajectory::GetConfigTrajectory() const {
        return q_;
    }

    SimpleTrajectory Trajectory::GetVelocityTrajectory() const {
        return v_;
    }
} //torc::mpc