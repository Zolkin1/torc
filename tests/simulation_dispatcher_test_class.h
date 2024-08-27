//
// Created by zolkin on 8/10/24.
//

#ifndef SIMULATION_DISPATCHER_TEST_CLASS_H
#define SIMULATION_DISPATCHER_TEST_CLASS_H

#include <iostream>
#include <catch2/catch_test_macros.hpp>

#include "cost_function.h"

namespace torc::sample {
    class SimulationDispatcherTest : SimulationDispatcher {
    public:
        SimulationDispatcherTest(const fs::path& xml_path, int num_samples)
            : SimulationDispatcher(xml_path, num_samples) {}

        void CheckModelName(const std::string& expected_name) {
            CHECK(GetModelName() == expected_name);
        }

        void CheckActuatedJoints() {
            if (GetModelName() == "achilles") {
                CHECK(act_joint_id_.size() == 18);
            }
        }

        void CheckSingleSimuation(const torc::mpc::Trajectory& traj_ref) {
            mpc::Trajectory traj_out;
            traj_out.UpdateSizes(traj_ref.GetConfiguration(0).size(),
                traj_ref.GetVelocity(0).size(), traj_ref.GetTau(0).size(),
                traj_ref.GetContactFrames(), traj_ref.GetNumNodes());
            mpc::ContactSchedule cs_out;
            InputSamples samples;
            // Test with the same dt
            samples.dt = traj_ref.GetDtVec();
            samples.type = Position;

            const vectorx_t act_sample = vectorx_t::Constant(traj_ref.GetTau(0).size(), 1);
            for (int i = 0; i < traj_ref.GetNumNodes(); i++) {
                samples.InsertSample(i, act_sample);
            }

            SingleSimulation(samples, traj_ref, traj_out, cs_out);
        }

        void CheckBatchSimulation(const torc::mpc::Trajectory& traj_ref) {
            mpc::Trajectory traj_out;
            traj_out.UpdateSizes(traj_ref.GetConfiguration(0).size(),
                traj_ref.GetVelocity(0).size(), traj_ref.GetTau(0).size(),
                traj_ref.GetContactFrames(), traj_ref.GetNumNodes());
            mpc::ContactSchedule cs_out;
            InputSamples samples;
            // Test with the same dt
            samples.dt = traj_ref.GetDtVec();
            samples.type = Position;

            const vectorx_t act_sample = vectorx_t::Constant(traj_ref.GetTau(0).size(), 1);
            for (int i = 0; i < traj_ref.GetNumNodes(); i++) {
                samples.InsertSample(i, act_sample);
            }

            const int NUM_SAMPLES = data_.size();
            std::vector<mpc::Trajectory> trajectories(NUM_SAMPLES);
            std::fill(trajectories.begin(), trajectories.end(), traj_out);

            std::vector<mpc::ContactSchedule> contact_schedules(NUM_SAMPLES);
            for (auto& cs : contact_schedules) {
                cs = cs_out;
            }

            std::vector<InputSamples> samples_vec(NUM_SAMPLES);
            for (auto& s : samples_vec) {
                s = samples;
            }

            BatchSimulation(samples_vec, traj_ref, trajectories, contact_schedules);
        }

        void CheckQuaternionConversions() {
            torc::mpc::vectorx_t q_nom(7), q_mujoco(7);
            q_nom << 1, 2, 3, 4, 5, 6, 7;
            q_mujoco << 1, 2, 3, 7, 4, 5, 6;

            vectorx_t q_out = ChangeQuaternionConventionToMujoco(q_nom);
            CHECK(q_out == q_mujoco);

            vectorx_t q_out2 = ChangeQuaternionConventionFromMujoco(q_out);
            CHECK(q_out2 == q_nom);

        }

        void BenchmarkSims(const torc::mpc::Trajectory& traj_ref) {
            mpc::Trajectory traj_out;
            traj_out.UpdateSizes(traj_ref.GetConfiguration(0).size(),
                traj_ref.GetVelocity(0).size(), traj_ref.GetTau(0).size(),
                traj_ref.GetContactFrames(), traj_ref.GetNumNodes());
            mpc::ContactSchedule cs_out;
            InputSamples samples;
            // Test with the same dt
            samples.dt = traj_ref.GetDtVec();
            samples.type = Position;

            const vectorx_t act_sample = vectorx_t::Constant(traj_ref.GetTau(0).size(), 1);
            for (int i = 0; i < traj_ref.GetNumNodes(); i++) {
                samples.InsertSample(i, act_sample);
            }

            const int NUM_SAMPLES = data_.size();
            std::vector<mpc::Trajectory> trajectories(NUM_SAMPLES);
            std::fill(trajectories.begin(), trajectories.end(), traj_out);

            std::vector<mpc::ContactSchedule> contact_schedules(NUM_SAMPLES);
            for (auto& cs : contact_schedules) {
                cs = cs_out;
            }

            std::vector<InputSamples> samples_vec(NUM_SAMPLES);
            for (auto& s : samples_vec) {
                s = samples;
            }

            BENCHMARK("Single Sim") {
                SingleSimulation(samples, traj_ref, traj_out, cs_out);
            };

            BENCHMARK("Batch Sim") {
                BatchSimulation(samples_vec, traj_ref, trajectories, contact_schedules);
            };
        }
    protected:
    private:
    };
}

#endif //SIMULATION_DISPATCHER_TEST_CLASS_H
