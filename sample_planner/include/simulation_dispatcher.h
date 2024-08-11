//
// Created by zolkin on 8/8/24.
//

#ifndef SIMULATION_DISPATCHER_H
#define SIMULATION_DISPATCHER_H

#include <filesystem>
#include <vector>
#include <mujoco/mujoco.h>

#include "BS_thread_pool.hpp"

#include "trajectory.h"
#include "contact_schedule.h"

namespace torc::sample {
    using vectorx_t = Eigen::VectorXd;
    using vector3_t = Eigen::Vector3d;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3x_t = Eigen::Matrix3Xd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix43_t = Eigen::Matrix<double, 4, 3>;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    namespace fs = std::filesystem;

    enum SampleType {
        Position,
        Torque
    };

    struct InputSamples {
        std::vector<double> dt; // dt associated with the samples - can be different from the reference trajectory
        matrixx_t samples;      // sample node x actuator
        SampleType type;        // Type of sample
        std::map<std::string, int> actuator_to_idx; // TODO: Do this better: Maps the actuator name into the trajector vectors

        void InsertSample(int sample_node, const vectorx_t& actuator_sample);
    };

    class SimulationDispatcher {
    public:
        // Deals with the Mujoco interface and thread pool
        SimulationDispatcher(const fs::path& xml_path, int num_samples);
        SimulationDispatcher(const fs::path& xml_path, int num_samples, int num_threads);

        void SingleSimulation(const InputSamples& samples, const mpc::Trajectory& traj_ref, mpc::Trajectory& traj_out, mpc::ContactSchedule& cs_out);

        void BatchSimulation(const std::vector<InputSamples>& samples, const mpc::Trajectory& traj_ref, std::vector<mpc::Trajectory>& trajectories,
            std::vector<mpc::ContactSchedule>& contact_schedules);

        std::string GetModelName() const;

        /**
         *
         * @return the list of actuated joint names as used in all the torque vectors
         */
        std::vector<std::string> GetJointOrder() const;
    protected:
        void CreateMJModelData(const std::string& xml_path, int num_samples);

        /**
         *
         * @param samples the input samples (about zero) to apply
         * @param traj the reference trajectory which include the nominal inputs and the initial condition
         * @param cs_out the resulting contact schedule
         * @param robot_num the robot number (identifier) to use
         */
        void Simulate(const InputSamples& samples, const mpc::Trajectory& traj_ref, mpc::Trajectory& traj_out, mpc::ContactSchedule& cs_out, int robot_num);

        /**
         *
         * @param data the mujoco data to reset
         * @param traj the trajectory to use for the initial condition
         */
        void ResetData(mjData* data, const mpc::Trajectory& traj) const;

        static int GetNode(const std::vector<double>& dts, double time);

        double GetTotalJointTorque(const mjData* data, int jnt_id) const;

        void VerifyTrajectory(const mpc::Trajectory& traj) const;

        mjModel* model_; // MuJoCo model
        std::vector<mjData*> data_; // MuJoCo data

        std::vector<int> act_joint_id_;    // ID of all the actuated joints, in the same order as tau

        BS::thread_pool pool;   // Thread pool

    private:
    };
} // namespace torc::sample


#endif //SIMULATION_DISPATCHER_H
