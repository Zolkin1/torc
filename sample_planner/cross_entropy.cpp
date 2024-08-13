//
// Created by zolkin on 8/7/24.
//
#include "yaml-cpp/yaml.h"

#include "cross_entropy.h"

namespace torc::sample {
    CrossEntropy::CrossEntropy(const std::filesystem::path& xml_path, int num_samples,
        const std::filesystem::path& config_path)
        : dispatcher_(xml_path, num_samples), num_samples_(num_samples) {
        // Read in the configs
        YAML::Node config;
        try {
            config = YAML::LoadFile(config_path);
        } catch (...) {
            throw std::runtime_error("Could not load the configuration file!");
        }

        // ---------- General Settings ---------- //
        if (!config["cem_settings"]) {
            throw std::runtime_error("No cem_settings provided!");
        }

        YAML::Node cem_settings = config["cem_settings"];
        if (cem_settings["sample_variance"]) {
            sample_variance_ = cem_settings["sample_variance"].as<double>();
        } else {
            throw std::runtime_error("Sample variance not specified!");
        }

        if (cem_settings["num_finalists"]) {
            num_finalists_ = cem_settings["num_finalists"].as<int>();
        } else {
            throw std::runtime_error("Number of finalists not specified!");
        }

        if (cem_settings["actuator_names"]) {
            actuators_ = cem_settings["actuator_names"].as<std::vector<std::string>>();
        } else {
            throw std::runtime_error("Actuator names not specified!");
        }

        if (cem_settings["sample_type"]) {
            std::string type = cem_settings["sample_type"].as<std::string>();
            if (type == "Position") {
                sample_type_ = Position;
            } else if (type == "Torque") {
                sample_type_ = Torque;
            } else {
                throw std::runtime_error("Provided sample type is not supported!");
            }
        } else {
            throw std::runtime_error("No sample type provided!");
        }

        // Size everything
        trajectories_.resize(num_samples_);
        contact_schedules_.resize(num_samples);
        samples_.resize(num_samples);
    }

    void CrossEntropy::Plan(const mpc::Trajectory &traj_ref, mpc::Trajectory &traj_out) {
        for (int i = 0; i < num_samples_; i++) {
            trajectories_[i].UpdateSizes(traj_ref.GetConfiguration(0).size(),
                traj_ref.GetVelocity(0).size(), traj_ref.GetTau(0).size(),
                traj_ref.GetContactFrames(), traj_ref.GetNumNodes());
            samples_[i].dt = traj_ref.GetDtVec();
            samples_[i].type = sample_type_;

            // TODO: Fill out the actuator name to index map!

            // For now I won't multithread the sampling, but in the future I may want to
            vectorx_t actuator_sample;
            for (int node = 0; node < traj_ref.GetNumNodes(); node++) {
                GetActuatorSamples(actuator_sample);
                samples_[i].InsertSample(node, actuator_sample);
            }
        }

        // TODO: Look into making sure this is called correctly.
        dispatcher_.BatchSimulation(samples_, traj_ref, trajectories_, contact_schedules_);

        // TODO: Analyze the results
    }

    void CrossEntropy::GetActuatorSamples(vectorx_t &actuator_sample) {
        actuator_sample.resize(actuators_.size());
        // *** Note *** if I multithread this function then each thread will need a bit_gen bc its not thread safe!
        for (int i = 0; i < actuators_.size(); i++) {
            // Get a normally distrubted value (taken with mean 0 since it will be applied about the reference trajectory)
            actuator_sample(i) = absl::Gaussian(bit_gen_, 0.0, sample_variance_);
        }
    }



} // namespace torc::sample