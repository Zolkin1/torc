//
// Created by zolkin on 8/7/24.
//

#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include <filesystem>
#include "absl/random/random.h"

#include "full_order_mpc.h"
#include "simulation_dispatcher.h"

namespace torc::sample {
    class CrossEntropy {
    public:
        // Read in the mujoco XML
        // Create one model and many data
        // Create the thread pool
        // Get the current trajectory
        // Sample inputs (positions, torques, etc...) about 0
        // Pass the samples and current trajectory to be simulated
        // Evaluate the trajectories based on a cost function
        // Weight top N
        // Take weighted average
        // Re-simulate average to get the final contact schedule
        CrossEntropy(const std::filesystem::path& xml_path, int num_samples, const std::filesystem::path& config_path,
                     std::shared_ptr<torc::mpc::FullOrderMpc> mpc, unsigned int num_threads = std::thread::hardware_concurrency());

        /**
         * @brief uses CEM to plan a trajectory about a reference trajectory
         * @param traj_ref the reference trajectory
         * @param traj_out the output trajectory
         * @param cs_out the resulting contact schedule
         * @param cost_types the cost names to use to evaluate the performance of the trajectory
         */
        void Plan(const mpc::Trajectory& traj_ref, mpc::Trajectory& traj_out, mpc::ContactSchedule& cs_out,
                  const std::vector<std::string>& cost_names);

        double GetVariance() const;
        double GetAvgCost() const;
    protected:
        void GetActuatorSamples(vectorx_t& actuator_sample);

        void ResizeSamples(const mpc::Trajectory& traj_ref);

        void UpdateCostAvg(const std::map<double, int>& costs);
        void UpdateCostVariance(const std::map<double, int>& costs);

    private:
        SimulationDispatcher dispatcher_;

        double sample_variance_;
        double cost_avg_;
        int num_finalists_;
        std::vector<std::pair<int, std::array<std::string, 3>>> idx_to_actuator_;
        SampleType sample_type_;
        int num_samples_;

        // Intermediate values
        std::vector<mpc::Trajectory> trajectories_;
        std::vector<mpc::ContactSchedule> contact_schedules_;
        std::vector<InputSamples> samples_;

        mpc::Trajectory sum_traj_;

        // Random library
        std::default_random_engine generator_;

        // Cost function
        std::shared_ptr<torc::mpc::FullOrderMpc> mpc_;
    };
} // namespace torc::sample


#endif //CROSSENTROPY_H
