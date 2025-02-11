//
// Created by zolkin on 6/18/24.
//

#ifndef TORC_TRAJECTORY_H
#define TORC_TRAJECTORY_H

#include <map>
#include <optional>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "simple_trajectory.h"

namespace torc::mpc {
    using vectorx_t = Eigen::VectorXd;
    using vector3_t = Eigen::Vector3d;
    using quat_t = Eigen::Quaterniond;

    class Trajectory {
    public:
        Trajectory();

        void UpdateSizes(int config_size, int vel_size, int tau_size, const std::vector<std::string>& force_frames, int nodes);

        int GetNumNodes() const;
        void SetNumNodes(int nodes);

        void SetConfiguration(int node, const vectorx_t& q);
        void SetVelocity(int node, const vectorx_t& v);
        void SetTau(int node, const vectorx_t& tau);
        void SetForce(int node, const std::string& frame, const vector3_t& f);
        void SetDtVector(const std::vector<double>& dt);
        void SetInContact(int node, const std::string &frame, bool in_contact);

        [[nodiscard]] vectorx_t GetConfiguration(int node) const;
        [[nodiscard]] quat_t GetQuat(int node) const;
        [[nodiscard]] vectorx_t GetVelocity(int node) const;
        [[nodiscard]] vectorx_t GetTau(int node) const;
        [[nodiscard]] vector3_t GetForce(int node, const std::string& frame) const ;
        [[nodiscard]] const std::vector<double>& GetDtVec() const;
        [[nodiscard]] std::vector<std::string> GetContactFrames() const;
        [[nodiscard]] double GetTotalTime() const;
        [[nodiscard]] bool GetInContact(const std::string& frame, int node) const;

        void GetConfigInterp(double time, vectorx_t& q_out);
        void GetVelocityInterp(double time, vectorx_t& v_out);
        void GetTorqueInterp(double time, vectorx_t& torque_out);
        void GetForceInterp(double time, const std::string& frame, vector3_t& force_out);
        bool GetInContactInterp(double time, const std::string& frame);

        void SetDefault(const vectorx_t& q_default);

        void ExportToCSV(const std::string& file_path);

        SimpleTrajectory GetConfigTrajectory() const;
        SimpleTrajectory GetVelocityTrajectory() const;

    protected:
    private:
        void StandardVectorInterp(double time, vectorx_t& vec_out, const SimpleTrajectory& vecs);

        [[nodiscard]] std::optional<int> GetNode(double time) const;

        static constexpr int POS_VARS = 3;
        static constexpr int QUAT_VARS = 4;
        static constexpr int FLOATING_BASE = 7;


        SimpleTrajectory q_;
        SimpleTrajectory v_;
        SimpleTrajectory tau_;
        std::vector<std::vector<vector3_t>> forces_;
        std::map<std::string, int> force_frames_;
        std::vector<double> dt_;
        std::vector<std::vector<bool>> in_contact_;
        int num_frames_;
        int nodes_;
        int config_size_;
        int vel_size_;
        int tau_size_;
    };
}

#endif //TORC_TRAJECTORY_H
