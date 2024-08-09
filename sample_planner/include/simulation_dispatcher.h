//
// Created by zolkin on 8/8/24.
//

#ifndef SIMULATION_DISPATCHER_H
#define SIMULATION_DISPATCHER_H

#include <vector>
#include <mujoco/mujoco.h>

#include "BS_thread_pool.hpp"

#include "trajectory.h"
#include "contact_schedule.h"

namespace torc::sample {
    using vectorx_t = Eigen::VectorXd;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3x_t = Eigen::Matrix3Xd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix43_t = Eigen::Matrix<double, 4, 3>;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;

    enum SampleType {
        Position,
        Torque
    };

    struct InputSamples {
        std::vector<double> dt;
        int num_samples;
        matrixx_t samples;
        SampleType type;
        std::vector<std::string> actuator_names;
        // TODO: Do this better:
        std::map<std::string, int> actuator_to_idx; // Maps the actuator name into the trajector vectors
    };

    class SimulationDispatcher {
    public:
        // Deals with the Mujoco interface and thread pool
        SimulationDispatcher(const std::string& xml_path, int num_samples);
        SimulationDispatcher(const std::string& xml_path, int num_samples, int num_threads);

        void SingleSimulation(const InputSamples& samples, const mpc::Trajectory& traj_ref, mpc::Trajectory& traj_out, mpc::ContactSchedule& cs_out);

        void BatchSimulation(const std::vector<InputSamples>& samples, const mpc::Trajectory& traj_ref, std::vector<mpc::Trajectory>& trajectories,
            std::vector<mpc::ContactSchedule>& contact_schedules);
    protected:
    private:
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

        int GetNode(const std::vector<double>& dts, double time);

        int GetPositionActuatorCount();
        int GetMotorActuatorCount();

    mjModel* model_; // MuJoCo model
    std::vector<mjData*> data_; // MuJoCo data

    BS::thread_pool pool;   // Thread pool

    };
} // namespace torc::sample


#endif //SIMULATION_DISPATCHER_H
