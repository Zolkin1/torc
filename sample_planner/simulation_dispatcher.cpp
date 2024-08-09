//
// Created by zolkin on 8/8/24.
//

#include <filesystem>

#include "simulation_dispatcher.h"

#include <iostream>


namespace torc::sample {
    // TODO: How to deal with errors without throwing?
    SimulationDispatcher::SimulationDispatcher(const std::string& xml_path, int num_samples)
        : model_(nullptr) {
        CreateMJModelData(xml_path, num_samples);
    }

    SimulationDispatcher::SimulationDispatcher(const std::string &xml_path, int num_samples, int num_threads)
        : model_(nullptr), pool(num_threads) {
        CreateMJModelData(xml_path, num_samples);
    }


    void SimulationDispatcher::SingleSimulation(const InputSamples &samples, const mpc::Trajectory &traj_ref, mpc::Trajectory& traj_out, mpc::ContactSchedule& cs_out) {
        Simulate(samples, traj_ref, traj_out, cs_out, 0);
    }

    void SimulationDispatcher::BatchSimulation(const std::vector<InputSamples> &samples, const mpc::Trajectory& traj_ref,
        std::vector<mpc::Trajectory> &trajectories, std::vector<mpc::ContactSchedule>& contact_schedules) {

        if (samples.size() != trajectories.size() || samples.size() != contact_schedules.size()) {
            throw std::invalid_argument("[Simulation Dispatcher] Sample, trajectory, and contact schedule sizes don't match!");
        }

        BS::multi_future<void> loop_future = pool.submit_loop<size_t>(0, samples.size(),
            [this, samples, &trajectories, &contact_schedules, &traj_ref](size_t i) {
            this->Simulate(samples[i], traj_ref, trajectories[i], contact_schedules[i], i);
        });

        loop_future.wait();
    }


    void SimulationDispatcher::Simulate(const InputSamples &samples, const mpc::Trajectory &traj_ref, mpc::Trajectory& traj_out, mpc::ContactSchedule& cs_out, int robot_num) {
        double total_time = 0;
        for (const auto& delta : samples.dt) {
            total_time += delta;
        }

        if (traj_ref.GetNumNodes() <= 0) {
            throw std::runtime_error("Provided trajectory must have a positive number of nodes!");
        }

        if (samples.type == Position) {
            if (samples.samples.cols() != GetPositionActuatorCount()) {
                throw std::runtime_error("Sample inputs does not match the number of position actuators!");
            }
        }

        if (samples.type == Torque) {
            if (samples.samples.cols() != GetMotorActuatorCount()) {
                throw std::runtime_error("Sample inputs does not match the number of motor actuators!");
            }
        }

        if (samples.actuator_names.size() != samples.samples.cols()) {
            throw std::runtime_error("Number of named actuators does not match the number of actuator samples!");
        }

        // Clear data from previous sims
        ResetData(data_[robot_num], traj_ref);

        // Simulate
        while (data_[robot_num]->time <= total_time) {
            int sample_node = GetNode(samples.dt, data_[robot_num]->time);
            if (sample_node == -1) {
                std::cerr << "Sample node limit hit early. Breaking." << std::endl;
                break;
            }

            int traj_node = GetNode(traj_ref.GetDtVec(), data_[robot_num]->time);
            if (traj_node == -1) {
                std::cerr << "Trajectory node limit hit early. Breaking." << std::endl;
                break;
            }

            for (int i = 0; i < samples.actuator_names.size(); i++) {
                int actuator_id = mj_name2id(model_, mjOBJ_ACTUATOR, samples.actuator_names[i].c_str());
                if (actuator_id != -1) {
                    data_[robot_num]->ctrl[actuator_id] = samples.samples(sample_node, i);
                    if (samples.type == Position) {
                        // TODO: Add velocity targets for position too
                        data_[robot_num]->ctrl[actuator_id] += traj_ref.GetConfiguration(traj_node)(samples.actuator_to_idx.at(samples.actuator_names[i]));
                    } else if (samples.type == Torque) {
                        data_[robot_num]->ctrl[actuator_id] += traj_ref.GetTau(traj_node)(samples.actuator_to_idx.at(samples.actuator_names[i]));
                    }
                } else {
                    throw std::runtime_error("Provided actuator is not a valid Mujoco actuator.");
                }

                mj_step(model_, data_[robot_num]);

                // TODO: Start adding the data back to the trajectory
                // TODO: add to the contact schedule
                // Go through all the frames in the contact schedule and check their collisions.
                // Need to consider what they are colliding with
            }
        }
    }

    void SimulationDispatcher::ResetData(mjData *data, const mpc::Trajectory& traj) const {
        data->time = 0;
        for (int i = 0; i < model_->nq; i++) {
            data->qpos[i] = traj.GetConfiguration(0)[i];
        }

        for (int i = 0; i < model_->nv; i++) {
            data->qvel[i] = traj.GetVelocity(0)[i];
        }

        // For now, set the acceleration to 0
        std::memset(data->qpos, 0, model_->na * sizeof(mjtNum));

        // Set control to 0
        std::memset(data->ctrl, 0, model_->nu * sizeof(mjtNum));
    }

    int SimulationDispatcher::GetNode(const std::vector<double> &dts, double time) {
        double time_tally = dts[0];
        for (int i = 1; i < dts.size(); i++) {
            if (time < time_tally) {
                return i - 1;
            }

            time_tally += dts[i];
        }

        return -1;
    }


    // TODO: Check
    int SimulationDispatcher::GetPositionActuatorCount() {
        int positionActuatorCount = 0;

        for (int i = 0; i < model_->nu; ++i) {
            if (model_->actuator_gaintype[i] == mjGAIN_FIXED &&
                model_->actuator_biastype[i] == mjBIAS_AFFINE) {
                positionActuatorCount++;
            }
        }

        return positionActuatorCount;
    }

    // TODO: Check
    int SimulationDispatcher::GetMotorActuatorCount() {
        int torqueActuatorCount = 0;

        // Iterate through all actuators
        for (int i = 0; i < model_->nu; ++i) {
            // Check if the actuator is a motor type
            if (model_->actuator_dyntype[i] == mjDYN_NONE &&
                model_->actuator_gaintype[i] == mjGAIN_FIXED &&
                model_->actuator_biastype[i] == mjBIAS_NONE) {
                torqueActuatorCount++;
            }
        }

        return torqueActuatorCount;
    }


    void SimulationDispatcher::CreateMJModelData(const std::string &xml_path, int num_samples) {
        // load the mujoco model
        char error[1000] = "Could not load binary model";
        if (!std::filesystem::exists(std::filesystem::path(xml_path))) {
            throw std::invalid_argument("Mujoco XML path does not exist!");
        }

        model_ = mj_loadXML(xml_path.c_str(), 0, error, 1000);
        if (!model_) {
            throw std::runtime_error("Could not create Mujoco model!");
        }

        // Make the data for each sample
        data_.resize(num_samples);
        for (int i = 0; i < num_samples; i++) {
            data_[i] = mj_makeData(model_);
        }
    }


} // namespace torc::sample