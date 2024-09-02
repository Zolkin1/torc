//
// Created by zolkin on 8/8/24.
//

#include <filesystem>
#include <iostream>
#include <unordered_set>

#include "simulation_dispatcher.h"



namespace torc::sample {
    void InputSamples::InsertSample(int sample_node, const vectorx_t& actuator_sample) {
        samples.row(sample_node) = actuator_sample.transpose();
    }

    double InputSamples::GetSplineInterp(double time, int idx) const {
        // Determine which nodes to use as control points
        double node_time = 0;
        int upper_dt_idx = -1;
        for (int i = 0; i < dt.size(); i++) {
            if (time >= node_time && time < node_time + dt[i]) {
                upper_dt_idx = i;
                break;
            }
            node_time += dt[i];
        }

        if (upper_dt_idx == -1) {
            std::cerr << "Time: " << time << std::endl;
            std::cerr << "[Sample] Trajectory interpolation not at a valid time!" << std::endl;
            throw std::runtime_error("[Sample] Trajectory interpolation not at a valid time!");
        }

        // Interpolate between them - linear
        double tau = (time - node_time)/(dt[upper_dt_idx]);

        if (upper_dt_idx == 0) {
            return tau*samples(upper_dt_idx, idx);
        } else {
            return samples(upper_dt_idx - 1, idx)*(1-tau) + tau*samples(upper_dt_idx, idx);
        }
    }

    // TODO: How to deal with errors without throwing?
    SimulationDispatcher::SimulationDispatcher(const fs::path& xml_path, int num_samples)
        : model_(nullptr) {
        CreateMJModelData(xml_path, num_samples);
    }

    SimulationDispatcher::SimulationDispatcher(const fs::path& xml_path, int num_samples, int num_threads)
        : model_(nullptr), pool(num_threads) {
        CreateMJModelData(xml_path, num_samples);
    }


    void SimulationDispatcher::SingleSimulation(const InputSamples &samples, const mpc::Trajectory &traj_ref, mpc::Trajectory& traj_out, mpc::ContactSchedule& cs_out) {
        VerifyTrajectory(traj_ref);

        if (samples.idx_to_actuator.size() != samples.samples.cols()) {
            std::cerr << "actuator to idx map size: " << samples.idx_to_actuator.size() << std::endl;
            std::cerr << "sample cols: " << samples.samples.cols() << std::endl;
            throw std::runtime_error("[Simulation Dispatcher] Number of named actuators does not match the number of actuator samples!");
        }


        Simulate(samples, traj_ref, traj_out, cs_out, 0);
    }

    void SimulationDispatcher::BatchSimulation(const std::vector<InputSamples> &samples, const mpc::Trajectory& traj_ref,
        std::vector<mpc::Trajectory>& trajectories, std::vector<mpc::ContactSchedule>& contact_schedules) {
        VerifyTrajectory(traj_ref);

        if (samples.size() != trajectories.size() || samples.size() != contact_schedules.size()) {
            throw std::invalid_argument("[Simulation Dispatcher] Sample, trajectory, and contact schedule sizes don't match!");
        }

        for (const auto& sample : samples) {
            if (sample.idx_to_actuator.size() != sample.samples.cols()) {
                std::cerr << "number of actuators: " << sample.idx_to_actuator.size() << std::endl;
                std::cerr << "sample cols: " << sample.samples.cols() << std::endl;
                throw std::runtime_error("[Simulation Dispatcher] Number of named actuators does not match the number of actuator samples!");
            }
        }

        if (traj_ref.GetNumNodes() <= 0) {
            throw std::runtime_error("[Simulation Dispatcher] Provided trajectory must have a positive number of nodes!");
        }

        // TODO: Fix it so errors being thrown in the other threads are still visible
        BS::multi_future<void> loop_future = pool.submit_loop<size_t>(0, samples.size(),
            [this, &samples, &trajectories, &contact_schedules, &traj_ref](size_t i) {
            this->Simulate(samples[i], traj_ref, trajectories[i], contact_schedules[i], i);
        });

        loop_future.wait();
    }

    std::string SimulationDispatcher::GetModelName() const {
        return model_->names;
    }


    void SimulationDispatcher::Simulate(const InputSamples &samples, const mpc::Trajectory &traj_ref, mpc::Trajectory& traj_out, mpc::ContactSchedule& cs_out, int robot_num) {

        // TODO:
        //  - Move samples to a spline interpolation (non zero)
        //  - Calculate derivative based on the spline interpolation
        //  - Project sampled values into reasonable bounds (maybe)
        //  - Make sure the initial condition stays constant (i.e. is not modified by the sampling and sim)

        double total_time = 0;
        for (const auto& delta : samples.dt) {
            total_time += delta;
        }

        total_time = std::min(total_time, traj_ref.GetTotalTime());

        // Setup the output trajectory
        if (traj_out.GetNumNodes() != traj_ref.GetNumNodes()) {
            traj_out.SetNumNodes(traj_ref.GetNumNodes());
        }
        traj_out.SetDtVector(traj_ref.GetDtVec());
        // Assign initial condition
        traj_out.SetConfiguration(0, traj_ref.GetConfiguration(0));
        traj_out.SetVelocity(0, traj_ref.GetVelocity(0));
        traj_out.SetTau(0, traj_ref.GetTau(0));
        for (const auto& frame : traj_ref.GetContactFrames()) {
            traj_out.SetForce(0, frame, traj_ref.GetForce(0, frame));
        }

        // Set up the contact schedule
        cs_out.SetFrames(traj_ref.GetContactFrames());
        cs_out.Reset();

        // Clear data from previous sims
        ResetData(data_[robot_num], traj_ref);

        std::vector<double> contact_start_times(traj_ref.GetContactFrames().size());
        std::fill(contact_start_times.begin(), contact_start_times.end(), -1);

        std::map<std::string, vector3_t> f_avg;
        vectorx_t q_total = vectorx_t::Zero(model_->nq);        // Mujoco quaternion ordering
        vectorx_t v_total = vectorx_t::Zero(model_->nv);
        vectorx_t tau_total = vectorx_t::Zero(act_joint_id_.size());
        for (const auto& frame : traj_ref.GetContactFrames()) {
            f_avg[frame] = vector3_t::Zero();
        }
        int current_node = 1;
        int num_time_steps = 0;

        // Simulate
        while (data_[robot_num]->time < total_time) {
            int traj_node = GetNode(traj_ref.GetDtVec(), data_[robot_num]->time);
            if (traj_node == -1) {
                std::cerr << "Trajectory node limit hit early. Breaking." << std::endl;
                break;
            }

            if (traj_node != current_node && traj_node > 0) {
                if (current_node == 0 ) {
                    std::cerr << "CURRENT NODE = 0" << std::endl;
                }
                // Change quaternion convention coming out of Mujoco
                vectorx_t q = ChangeQuaternionConventionFromMujoco(q_total / num_time_steps);
                traj_out.SetConfiguration(current_node, q);


//                std::cerr << "-------- Current node: " << current_node << std::endl;
//                std::cerr << "traj ref node 0: " << traj_ref.GetConfiguration(0).transpose() << std::endl;
//                std::cerr << "q total: " << q_total.transpose()/num_time_steps << std::endl;
//                std::cerr << "num time steps: " << num_time_steps << std::endl;

                // Rotate into the local frame
                matrix3_t R = static_cast<torc::mpc::quat_t>((q).segment<4>(3)).toRotationMatrix();
                vectorx_t v = v_total/num_time_steps;
                v.head<3>() = R*v.head<3>();
                v.segment<3>(3) = R*v.segment<3>(3);

                traj_out.SetVelocity(current_node, v);
                traj_out.SetTau(current_node, tau_total/num_time_steps);
                for (const auto& frame : traj_ref.GetContactFrames()) {
                    traj_out.SetForce(current_node, frame, f_avg[frame]/num_time_steps);
                    f_avg[frame].setZero();
                }

                current_node++;
                num_time_steps = 0;
                q_total.setZero();
                v_total.setZero();
                tau_total.setZero();
            }

            // Check and update contact status
            for (const auto & geom : traj_ref.GetContactFrames()) {
                // *** Note *** The way this is currently done, it assumes that the geom will only ever be in contact with
                //  one object at a time. If it starts to be in contact with two objects at the same time, then it won't work
                //  to deal with multiple contacts I will need to track what it is in contact with.

                // For now I will not record what we are in contact with, but I can add that later
                const int geom1_idx = mj_name2id(model_, mjOBJ_GEOM, geom.c_str());
                // const int geom2_idx = model_->pair_geom2[idx];
                for (int  i = 0; i < data_[robot_num]->ncon; i++) {
                    if (geom1_idx == data_[robot_num]->contact[i].geom[0]) {
                        if (data_[robot_num]->contact[i].dist <= 0 && contact_start_times[i] < 0) {
                            // Contact is just starting bc start time is negative and dist is non-positive
                            contact_start_times[i] = data_[robot_num]->time;
                        } else if (data_[robot_num]->contact[i].dist > 0 && contact_start_times[i] >= 0) {
                            // Contact is ending bc the contact dist is positive and there was a recorded start time
                            throw std::runtime_error("TODO: Reimplement the contacts in the sample planner!");
                            // cs_out.InsertContact(mj_id2name(model_, mjOBJ_GEOM, geom1_idx), contact_start_times[i], data_[robot_num]->time);

                            // Reset the start time
                            contact_start_times[i] = -1;
                        }
                    }
                }
            }

            int sample_col = 0;

            if (samples.type == Position) {
                // Sample positions, finite difference velocities
                for (const auto& [idx, names] : samples.idx_to_actuator) {
                    int pos_actuator_id = mj_name2id(model_, mjOBJ_ACTUATOR, names[0].c_str());
                    int vel_actuator_id = mj_name2id(model_, mjOBJ_ACTUATOR, names[1].c_str());

                    if (pos_actuator_id != -1) {
                        double curr_targ_pos = samples.GetSplineInterp(data_[robot_num]->time, sample_col) +
                                traj_ref.GetConfiguration(traj_node)(idx);

                        // Always assume that at the next instant we are on the same traj node, not exactly true, but a good approx.
                        constexpr double FD_DELTA = 1e-8;
                        double next_targ_pos = samples.GetSplineInterp(data_[robot_num]->time, sample_col) +
                                traj_ref.GetConfiguration(traj_node)(idx); // + FD_DELTA

                        data_[robot_num]->ctrl[pos_actuator_id] = curr_targ_pos;
                        data_[robot_num]->ctrl[vel_actuator_id] = (next_targ_pos - curr_targ_pos)/FD_DELTA;
                    } else {
                        throw std::runtime_error("[Simulation Dispatcher] Provided actuator is not a valid Mujoco actuator.");
                    }
                    sample_col++;
                }
            } else if (samples.type == Torque) {
                // Sample torques
                for (const auto& [idx, names] : samples.idx_to_actuator) {
                    int actuator_id = mj_name2id(model_, mjOBJ_ACTUATOR, names[2].c_str());
                    if (actuator_id != -1) {
                        data_[robot_num]->ctrl[actuator_id] = samples.GetSplineInterp(data_[robot_num]->time, sample_col) + traj_ref.GetTau(traj_node)(idx);
                    } else {
                        throw std::runtime_error("[Simulation Dispatcher] Provided actuator is not a valid Mujoco actuator.");
                    }
                    sample_col++;
                }
            }

            mj_step(model_, data_[robot_num]);

            // Here we only add to the average
            if (traj_node == current_node && traj_node > 0) {
                num_time_steps++;

                Eigen::Map<Eigen::VectorXd> mj_vec_q(data_[robot_num]->qpos, model_->nq);
                q_total += mj_vec_q;

                Eigen::Map<Eigen::VectorXd> mj_vec_v(data_[robot_num]->qpos, model_->nq);
                v_total += mj_vec_v;

                // To get the torque I will need to sum all of the actuators on each joint
                for (int i = 0; i < tau_total.size(); i++) {
                    tau_total(i) += GetTotalJointTorque(data_[robot_num], act_joint_id_[i]);
                }

                // Get the contact forces
                // TODO: Check that I want to use crfc_ext and not something else
                for (const auto& geom : traj_ref.GetContactFrames()) {
                    const int geom1_idx = mj_name2id(model_, mjOBJ_GEOM, geom.c_str());  // For now I am assuming the name of geom1 is the associated contact frame
                    // TODO: What quantity do I extract here?

                }
            }

        }

        // Assign to last node
        // Change quaternion convention coming out of Mujoco
        vectorx_t q = ChangeQuaternionConventionFromMujoco(q_total / num_time_steps);
        traj_out.SetConfiguration(current_node, q);

        // Rotate into the local frame
        matrix3_t R = static_cast<torc::mpc::quat_t>((q).segment<4>(3)).toRotationMatrix();
        vectorx_t v = v_total/num_time_steps;
        v.head<3>() = R*v.head<3>();
        v.segment<3>(3) = R*v.segment<3>(3);
        traj_out.SetVelocity(current_node, v);

        traj_out.SetTau(current_node, tau_total/num_time_steps);
        for (const auto& frame : traj_ref.GetContactFrames()) {
            traj_out.SetForce(current_node, frame, f_avg[frame]/num_time_steps);
            f_avg[frame].setZero();
        }

        if (traj_out.GetConfiguration(0) != traj_ref.GetConfiguration(0) || traj_out.GetVelocity(0) != traj_ref.GetVelocity(0)) {
            std::cerr << "Trajectory IC modified!" << std::endl;
        }

    }

    void SimulationDispatcher::ResetData(mjData *data, const mpc::Trajectory& traj) const {
        data->time = 0;

        vectorx_t q = ChangeQuaternionConventionToMujoco(traj.GetConfiguration(0));
        for (int i = 0; i < model_->nq; i++) {
            data->qpos[i] = q(i);
        }

        // Rotate into the world frame
        matrix3_t R = static_cast<torc::mpc::quat_t>(traj.GetConfiguration(0).segment<4>(3)).inverse().toRotationMatrix();
        vectorx_t v = traj.GetVelocity(0);
        v.head<3>() = R*v.head<3>();
        v.segment<3>(3) = R*v.segment<3>(3);

        for (int i = 0; i < model_->nv; i++) {
            data->qvel[i] = v(i);
        }

        // For now, set the acceleration to 0
        std::memset(data->qpos, 0, model_->na * sizeof(mjtNum));

        // Set control to 0
        std::memset(data->ctrl, 0, model_->nu * sizeof(mjtNum));
    }

    int SimulationDispatcher::GetNode(const std::vector<double> &dts, double time) {
        double time_tally = dts[0];
        for (int i = 1; i < dts.size(); ++i) {
            if (time <= time_tally) {
                return i - 1;
            }

            time_tally += dts[i];
        }

        if (time <= time_tally) {
            return dts.size() - 1;
        }


        return -1;
    }

    double SimulationDispatcher::GetTotalJointTorque(const mjData *data, int jnt_id) const {
        double total_torque = 0.0;

        // Iterate over all actuators
        for (int i = 0; i < model_->nu; ++i) {
            // Check if the actuator affects the specified joint
            if (model_->actuator_trnid[i * 2] == jnt_id) {
                // Add the actuator force/torque to the total torque
                total_torque += data->actuator_force[i];
            }
        }

        return total_torque;
    }

    void SimulationDispatcher::VerifyTrajectory(const mpc::Trajectory &traj) const {
        // TODO: Consider not throwing here
        if (traj.GetConfiguration(0).size() != model_->nq) {
            throw std::invalid_argument("[Simulation Dispatcher] Trajectory configuration size does not match mujoco model!");
        }

        if (traj.GetVelocity(0).size() != model_->nv) {
            throw std::invalid_argument("[Simulation Dispatcher] Trajectory velocity size does not match mujoco model!");
        }

        if (traj.GetTau(0).size() != act_joint_id_.size()) {
            throw std::invalid_argument("[Simulation Dispatcher] Trajectory torque size does not match mujoco model!");
        }

        const std::vector<std::string> frames = traj.GetContactFrames();
        for (const auto& frame : frames) {
            // Verify that the frame matches a mujoco geom
            if (mj_name2id(model_, mjOBJ_GEOM, frame.c_str()) == -1) {
                throw std::invalid_argument("[Simulation Dispatcher] Trajectory force frame does match a mujoco geom!");
            }
        }
    }

    vectorx_t SimulationDispatcher::ChangeQuaternionConventionFromMujoco(const vectorx_t& config) {
        vectorx_t q = config;
        double w = q(3);
        q(3) = q(4);
        q(4) = q(5);
        q(5) = q(6);
        q(6) = w;

        return q;
    }

    vectorx_t SimulationDispatcher::ChangeQuaternionConventionToMujoco(const torc::sample::vectorx_t& config) {
        vectorx_t q = config;
        double w = q(6);
        q(6) = q(5);
        q(5) = q(4);
        q(4) = q(3);
        q(3) = w;

        return q;
    }

    void SimulationDispatcher::CreateMJModelData(const std::string &xml_path, int num_samples) {
        // load the mujoco model
        char error[1000] = "Could not load binary model";
        if (!std::filesystem::exists(std::filesystem::path(xml_path))) {
            throw std::invalid_argument("[Simulation Dispatcher] Mujoco XML path does not exist!");
        }

        model_ = mj_loadXML(xml_path.c_str(), 0, error, 1000);
        if (!model_) {
            throw std::runtime_error("[Simulation Dispatcher] Could not create Mujoco model!");
        }

        // Make the data for each sample
        data_.resize(num_samples);
        for (int i = 0; i < num_samples; i++) {
            data_[i] = mj_makeData(model_);
        }

        // Create the list of actuated joint id's
        std::unordered_set<int> joint_ids;

        // Iterate over all actuators
        for (int i = 0; i < model_->nu; ++i) {
            int joint_id_actuator = model_->actuator_trnid[i * 2]; // actuator_trnid is a 2D array flattened to 1D

            // Add the joint ID to the set
            joint_ids.insert(joint_id_actuator);
        }

        // Convert the set to a vector
        act_joint_id_ = std::vector<int>(joint_ids.begin(), joint_ids.end());
    }

    // Destructor
    SimulationDispatcher::~SimulationDispatcher() {
        // free MuJoCo model and data
        for (auto& data : data_) {
            if (data) {
                mj_deleteData(data);
            }
        }
        mj_deleteModel(model_);
    }


} // namespace torc::sample