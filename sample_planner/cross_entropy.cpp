//
// Created by zolkin on 8/7/24.
//
#include "yaml-cpp/yaml.h"

#include "cross_entropy.h"

namespace torc::sample {
    CrossEntropy::CrossEntropy(const std::filesystem::path& xml_path, int num_samples,
        const std::filesystem::path& config_path, std::shared_ptr<torc::mpc::FullOrderMpc> mpc, unsigned int num_threads)
        : dispatcher_(xml_path, num_samples, num_threads), num_samples_(num_samples), mpc_(mpc),
        cost_avg_(0) {
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

        if (cem_settings["actuator_idxs"]) {
            const auto indexes = cem_settings["actuator_idxs"].as<std::vector<int>>();
            for (int index : indexes) {
                idx_to_actuator_.emplace_back(index, std::array<std::string, 3>());
            }
        } else {
            throw std::runtime_error("Actuator indexes not specified!");
        }

        if (cem_settings["pos_actuator_names"]) {
            const auto names = cem_settings["pos_actuator_names"].as<std::vector<std::string>>();
            for (int i = 0; i < names.size(); i++) {
                idx_to_actuator_[i].second[0] = names[i];
            }
        } else {
            throw std::runtime_error("Position actuator names not specified!");
        }

        if (cem_settings["vel_actuator_names"]) {
            const auto names = cem_settings["vel_actuator_names"].as<std::vector<std::string>>();
            for (int i = 0; i < names.size(); i++) {
                idx_to_actuator_[i].second[1] = names[i];
            }
        } else {
            throw std::runtime_error("Velocity actuator names not specified!");
        }

        if (cem_settings["torque_actuator_names"]) {
            const auto names = cem_settings["torque_actuator_names"].as<std::vector<std::string>>();
            for (int i = 0; i < names.size(); i++) {
                idx_to_actuator_[i].second[2] = names[i];
            }
        } else {
            throw std::runtime_error("Torque actuator names not specified!");
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

    void CrossEntropy::Plan(const mpc::Trajectory &traj_ref, mpc::Trajectory &traj_out,
                            mpc::ContactSchedule& cs_out, const std::vector<std::string>& cost_names) {
        ResizeSamples(traj_ref);
        for (int i = 0; i < num_samples_; i++) {
            trajectories_[i].UpdateSizes(traj_ref.GetConfiguration(0).size(),
                traj_ref.GetVelocity(0).size(), traj_ref.GetTau(0).size(),
                traj_ref.GetContactFrames(), traj_ref.GetNumNodes());
            samples_[i].dt = traj_ref.GetDtVec();
            samples_[i].type = sample_type_;
            // TODO: Maybe set this differently
            samples_[i].idx_to_actuator = idx_to_actuator_;

            // For now I won't multithread the sampling, but in the future I may want to
            vectorx_t actuator_sample;
            for (int node = 0; node < traj_ref.GetNumNodes(); node++) {
                GetActuatorSamples(actuator_sample);
                samples_[i].InsertSample(node, actuator_sample);
            }
        }

        sum_traj_ = traj_ref;


        dispatcher_.BatchSimulation(samples_, traj_ref, trajectories_, contact_schedules_);

        // Get the current cost snapshot
        auto cost_targets = mpc_->GetCostSnapShot();

        // Remove unwanted cost terms
        std::erase_if(cost_targets.cost_data, [&cost_names](const torc::mpc::CostData& data) {
            bool name_match = false;
            for (const auto& name : cost_names) {
                if (data.constraint_name == name) {
                    name_match = true;
                }
            }

            return !name_match;
        });

        std::map<double, int> costs;
        // TODO: Get the costs in the seperate thread each
        for (int i = 0; i < num_samples_; i++) {
//            std::cout << "Cost: " << mpc_.GetTrajCost(trajectories_[i], cost_targets) << std::endl;
            costs.emplace(mpc_->GetTrajCost(trajectories_[i], cost_targets), i);
        }

        // For updating the cem avg and variance
        UpdateCostAvg(costs);
        // TODO: Put back
//        UpdateCostVariance(costs);


            // Now get the top N and average them
        for (int i = 0; i < traj_ref.GetNumNodes(); i++) {
            vectorx_t q = vectorx_t::Zero(traj_ref.GetConfiguration(0).size());
            vectorx_t v = vectorx_t::Zero(traj_ref.GetVelocity(0).size());
            vectorx_t tau = vectorx_t::Zero(traj_ref.GetTau(0).size());
            std::map<std::string, vector3_t> f;
            for (const auto& frame : traj_ref.GetContactFrames()) {
                f.emplace(frame, vector3_t::Zero());
            }

            int count = 0;

            double prev_cost = -1;

            for (const auto& [cost, idx]: costs) {
                if (count == num_finalists_) {
                    break;
                }

                // TODO: Remove
                if (cost < prev_cost) {
                    std::cerr << "Cost ordering is wrong!" << std::endl;
                }
                prev_cost = cost;


                q = q + trajectories_[idx].GetConfiguration(i);
                v = v + trajectories_[idx].GetVelocity(i);
                tau = tau + trajectories_[idx].GetTau(i);

                for (const auto& frame : traj_ref.GetContactFrames()) {
                    f[frame] += trajectories_[idx].GetForce(i, frame);
                }

                count++;
            }

            sum_traj_.SetConfiguration(i, q/num_finalists_);
            sum_traj_.SetVelocity(i, v/num_finalists_);
            sum_traj_.SetTau(i, tau/num_finalists_);
            for (const auto& frame : traj_ref.GetContactFrames()) {
                sum_traj_.SetForce(i, frame, f[frame]/num_finalists_);
            }
        }


        samples_[0].samples.setZero();

        // TODO: Put back
//        dispatcher_.SingleSimulation(samples_[0], sum_traj_, traj_out, cs_out);

        traj_out = sum_traj_;

        // TODO: Why is the averaged trajectory so much worse after being simulated?
//        std::cout << "--- Sum traj Cost: " << mpc_.GetTrajCost(sum_traj_, cost_targets) << std::endl;
//        std::cout << "--- Final Cost: " << mpc_.GetTrajCost(traj_out, cost_targets) << std::endl;
//        std::cout << "--- Final Variance: " << GetVariance() << std::endl;
    }

    double CrossEntropy::GetVariance() const {
        return sample_variance_;
    }

    double CrossEntropy::GetAvgCost() const {
        return cost_avg_;
    }

    void CrossEntropy::UpdateCostAvg(const std::map<double, int>& costs) {
        cost_avg_ = 0;
        int count = 0;
        for (const auto& [cost, idx]: costs) {
            if (count == num_finalists_) {
                break;
            }

            cost_avg_ += cost;

            count++;
        }

        cost_avg_ = cost_avg_/num_finalists_;
    }

    void CrossEntropy::UpdateCostVariance(const std::map<double, int>& costs) {
        // Variance update
        sample_variance_ = 0;
        int count = 0;
        for (const auto& [cost, idx]: costs) {
            if (count == num_finalists_) {
                break;
            }

            sample_variance_ += std::pow(cost - cost_avg_, 2);

            count++;
        }

        std::cerr << "count: " << count << std::endl;

        sample_variance_ = sample_variance_/num_finalists_;
    }

    void CrossEntropy::GetActuatorSamples(vectorx_t& actuator_sample) {
        std::normal_distribution<double> normal_distr(0.0, std::sqrt(sample_variance_));
        actuator_sample.resize(idx_to_actuator_.size());
        // *** Note *** if I multithread this function then each thread will need a bit_gen bc its not thread safe!
        for (int i = 0; i < idx_to_actuator_.size(); i++) {
            // Get a normally distrubted value (taken with mean 0 since it will be applied about the reference trajectory)
            actuator_sample(i) = normal_distr(generator_);
        }
    }

    void CrossEntropy::ResizeSamples(const mpc::Trajectory& traj_ref) {
        for (auto& sample : samples_) {
            sample.samples.resize(traj_ref.GetNumNodes(), idx_to_actuator_.size());
        }
    }


} // namespace torc::sample