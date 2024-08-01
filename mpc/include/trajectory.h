//
// Created by zolkin on 6/18/24.
//

#ifndef TORC_TRAJECTORY_H
#define TORC_TRAJECTORY_H

#include <map>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace torc::mpc {
    using vectorx_t = Eigen::VectorXd;
    using vector3_t = Eigen::Vector3d;
    using quat_t = Eigen::Quaterniond;

    class Trajectory {
    public:
        void UpdateSizes(int config_size, int vel_size, int tau_size, const std::vector<std::string>& force_frames, int nodes);

        int GetNumNodes() const;
        void SetNumNodes(int nodes);

        void SetConfiguration(int node, const vectorx_t& q);
        void SetVelocity(int node, const vectorx_t& v);
        void SetTau(int node, const vectorx_t& tau);
        void SetForce(int node, const std::string& frame, const vector3_t& f);

        vectorx_t GetConfiguration(int node);
        quat_t GetQuat(int node);
        vectorx_t GetVelocity(int node);
        vectorx_t GetTau(int node);
        vector3_t GetForce(int node, const std::string& frame);

        void SetDefault(const vectorx_t& q_default);

//        void Reset();

        // TODO: Add interpolation function

    protected:
    private:
        std::vector<vectorx_t> q_;
        std::vector<vectorx_t> v_;
        std::vector<vectorx_t> tau_;
        std::vector<std::vector<vector3_t>> forces_;
        std::map<std::string, int> force_frames_;
        int num_frames_;
        int nodes_;
        int config_size_;
        int vel_size_;
        int tau_size_;
    };
}

#endif //TORC_TRAJECTORY_H
