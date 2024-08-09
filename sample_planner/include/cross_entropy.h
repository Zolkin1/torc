//
// Created by zolkin on 8/7/24.
//

#ifndef CROSSENTROPY_H
#define CROSSENTROPY_H

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
        CrossEntropy(const std::string& xml_path, int num_samples);
    protected:
    private:
        SimulationDispatcher dispatcher_;
    };
} // namespace torc::sample


#endif //CROSSENTROPY_H
