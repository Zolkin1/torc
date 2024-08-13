//
// Created by zolkin on 8/7/24.
//

#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

#include <filesystem>
#include "absl/random/random.h"

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
        CrossEntropy(const std::filesystem::path& xml_path, int num_samples, const std::filesystem::path& config_path);

        void Plan(const mpc::Trajectory& traj_ref, mpc::Trajectory& traj_out);
    protected:
        void GetActuatorSamples(vectorx_t& actuator_sample);

    private:
        SimulationDispatcher dispatcher_;

        double sample_variance_;
        int num_finalists_;
        std::vector<std::string> actuators_;
        SampleType sample_type_;
        int num_samples_;

        // Intermediate values
        std::vector<mpc::Trajectory> trajectories_;
        std::vector<mpc::ContactSchedule> contact_schedules_;
        std::vector<InputSamples> samples_;

        // Random library
        absl::BitGen bit_gen_;
    };
} // namespace torc::sample


#endif //CROSSENTROPY_H
