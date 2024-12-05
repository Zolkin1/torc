//
// Created by zolkin on 12/2/24.
//

#include "reference_generator.h"

namespace torc::mpc {
    ReferenceGenerator::ReferenceGenerator(int nodes, int config_size, int vel_size, std::vector<std::string> contact_frames,
                                           std::vector<double> dt, std::shared_ptr<models::FullOrderRigidBody> model)
    : nodes_(nodes), config_size_(config_size), vel_size_(vel_size), dt_(std::move(dt)), contact_frames_(contact_frames), model_(std::move(model)) {
        end_time_ = 0;
        for (const auto d : dt_) {
            end_time_ += d;
        }
    }

    // TODO: Consider modulating the commanded velocity and position dependent on the polytopes.
    //  e.g. if we are asked to walk off a polytope and no other polytope is selected then we should modulate the position
    //  and velocity to keep the body in the polytope, which will make the optimization easier. Just need to be careful
    //  that this is done correctly. Can also increase velocity/position if making ot over a big gap.
    std::pair<SimpleTrajectory, SimpleTrajectory> ReferenceGenerator::GenerateReference(const vectorx_t& q,
        const SimpleTrajectory& q_target,
        const SimpleTrajectory& v_target,
        const std::map<std::string, std::vector<double>>& swing_traj,
        const std::vector<double>& hip_offsets,
        const ContactSchedule& contact_schedule) {
        if (hip_offsets.size() != 2*contact_frames_.size()) {
            throw std::runtime_error("Hip offsets size != 2*contact_frames_.size()");
        }

        // NOTE: The commanded vel is assumed to be x-y-yaw

        // std::cout << "q: " << q.transpose() << std::endl;

        SimpleTrajectory q_ref(config_size_, nodes_);
        SimpleTrajectory v_ref(vel_size_, nodes_);

        std::map<std::string, std::vector<double>> contact_midtimes;
        std::vector<vector3_t> contact_base_positions;
        std::vector<vector3_t> ik_base_positions(nodes_);

        // First determine the nodes (and times) associated with the midpoints of the contacts
        const auto& contact_map = contact_schedule.GetScheduleMap();
        for (const auto& [frame, swings] : contact_map) {
            contact_midtimes.insert({frame, {}});

            if (!swings.empty()) {
                // Handle the contact midpoint before the first swing
                contact_midtimes[frame].emplace_back(swings[0].first - 0.15);
                // contact_midtimes[frame].emplace_back(std::max(0.0, swings[0].first - 0.15));
                for (int i = 0; i < swings.size() - 1; i++) {
                    contact_midtimes[frame].emplace_back((swings[i+1].first + swings[i].second)/2.0);
                }
                // Handle the contact midpoint after the last swing
                contact_midtimes[frame].emplace_back(swings[swings.size() - 1].second + 0.15);
            } else {
                contact_midtimes[frame].emplace_back(end_time_/2.0);
            }

            // TODO: Remove after debugging
            // std::cerr << "frame: " << frame << std::endl;
            for (int i = 1; i < contact_midtimes[frame].size(); i++) {
                // std::cout << "contact_midtimes[" << i << "]" << contact_midtimes[frame][i] << std::endl;
                if (contact_midtimes[frame][i] <= contact_midtimes[frame][i-1]) {
                    std::cerr << "i: " << i << std::endl;
                    std::cerr << "contact_midtimes[frame][i]: " << contact_midtimes[frame][i] << std::endl;
                    std::cerr << "contact_midtimes[frame][i-1]: " << contact_midtimes[frame][i-1] << std::endl;
                    throw std::runtime_error("[Reference Generator] contact midtimes not monotonically increasing!");
                }
                // std::cerr << "contact_midtimes[frame]["<< i <<"]: " << contact_midtimes[frame][i] << std::endl;
                // if (time > 10) {
                //     std::cerr << "time: " << contact_midtimes[frame][contact_midtimes[frame].size()-1] << std::endl;
                //     throw std::runtime_error("[RG] midtime too large!");
                // }
            }
        }

        // ------------------------------------------------------
        // Next, determine where the foot should be using the desired location and footstep constraints
        // ------------------------------------------------------
        // Holds the position of the feet at the contact midpoints for each frame
        std::map<std::string, std::vector<vector2_t>> contact_foot_pos;
        std::map<std::string, std::vector<vector2_t>> node_foot_pos;
        model_->FirstOrderFK(q);
        const auto& schedule = contact_schedule.GetScheduleMap();
        for (int j = 0; j < contact_frames_.size(); j++) {
            std::string& frame = contact_frames_[j];

            contact_foot_pos.insert({frame, {}});
            // contact_foot_pos[frame].resize(1);

            node_foot_pos.insert({frame, {}});

            // Current foot position
            // contact_foot_pos[frame][0] = model_->GetFrameState(frame).placement.translation().head<2>();

            vector2_t hip_offset;
            hip_offset << hip_offsets[2*j], hip_offsets[2*j + 1];

            // Determine all contact locations for the given frame
            for (int i = 0; i < contact_midtimes[frame].size(); i++) {
                // Get the contact location based on the midpoints and hip offsets
                // Get rotation matrix for the hip offsets

                if (contact_midtimes[frame][i] >= 0) {
                    vectorx_t q_command = GetCommandedConfig(contact_midtimes[frame][i], q_target, v_target);
                    const quat_t quat(q_command.segment<4>(3));
                    const matrix3_t R = quat.toRotationMatrix();

                    contact_foot_pos[frame].emplace_back(R.topLeftCorner<2,2>()*hip_offset
                        + q_command.head<2>());
                } else {
                    contact_midtimes[frame].erase(contact_midtimes[frame].begin() + i);
                    i--;
                }
            }

            // TODO: Remove after debugging
            // std::cout << "frame: " << frame << std::endl;
            // for (int i = 0; i < contact_foot_pos[frame].size(); i++) {
            //     std::cout << "foot pos: " << contact_foot_pos[frame][i].transpose() << " time: " << contact_midtimes[frame][i] << std::endl;
            // }

            int current_contact_idx = 0;
            for (int node = 0; node < nodes_; node++) {
                // std::cout << "node: " << node << " contact idx: " << current_contact_idx << std::endl;
                // Get the contact positions in front and behind the current time
                double time = GetTime(node);

                // if (contact_midtimes[frame][current_contact_idx] > time) {
                //     std::cout << "time: " << time << std::endl;
                //     std::cout << "current contact idx: " << current_contact_idx << std::endl;
                //     std::cout << "node: " << node << std::endl;
                // }

                // Check if we are in swing
                if (contact_schedule.InSwing(frame, time)) {
                    if (node > 0 && contact_schedule.InContact(frame, GetTime(node - 1))) {
                        current_contact_idx++;
                    }
                    // std::cerr << "[Reference Generator] midtimes size: " << contact_midtimes[frame].size() << std::endl;
                    // Get the index for the next contact
                    // int next_contact_idx = 0;
                    // while (contact_midtimes[frame].at(next_contact_idx) <= time) {
                    //     next_contact_idx++;
                    // }

                    if (current_contact_idx < 0) {
                        throw std::runtime_error("[Reference Generator] contact index issue! "
                                                 "Contact idx: " + std::to_string(current_contact_idx) + " time: " + std::to_string(time));
                    }

                    // TODO: Remove after debugging
                    // std::cerr << "next_contact_idx: " << next_contact_idx << std::endl;

                    // TODO: Remove after debugging
                    // std::cout << "swing duration: " << swing_duration << std::endl;
                    // std::cout << "swing start: " << swing_start << std::endl;
                    // std::cout << "lambda: " << lambda << std::endl;
                    // std::cout << "time: " << time << std::endl;
                    vector2_t swing_intermediate_pos;
                    if (current_contact_idx > 0) {
                        double swing_duration = contact_schedule.GetSwingDuration(frame, time);
                        double swing_start = contact_schedule.GetSwingStartTime(frame, time);
                        double lambda = (time - swing_start)/(swing_duration);

                        if (lambda > 1 || lambda < 0) {
                            throw std::runtime_error("[Reference Generator] lambda issue!");
                        }

                        swing_intermediate_pos = lambda*contact_foot_pos[frame].at(current_contact_idx)
                            + (1-lambda)*contact_foot_pos[frame][current_contact_idx-1];
                    } else if (current_contact_idx == 0) {
                        double swing_duration = contact_schedule.GetFirstContactTime(frame);
                        double swing_start = 0;
                        double lambda = (time - swing_start)/(swing_duration);

                        swing_intermediate_pos = lambda*contact_foot_pos[frame][current_contact_idx]
                            + (1-lambda)*model_->GetFrameState(frame).placement.translation().head<2>();
                    }

                    node_foot_pos[frame].emplace_back(swing_intermediate_pos);
                } else {
                    if (!contact_schedule.InContact(frame, GetTime(node))) {
                        throw std::runtime_error("[Reference Generator] should be in contact!");
                    }

                    if (node == 0) {
                        contact_foot_pos[frame].at(current_contact_idx) = model_->GetFrameState(frame).placement.translation().head<2>();
                    }

                    node_foot_pos[frame].emplace_back(contact_foot_pos[frame].at(current_contact_idx));

                    // if (contact_midtimes[frame][current_contact_idx] > end_time_) {
                    //     std::cerr << "end time: " << end_time_ << std::endl;
                    //     std::cerr << "node: " << node << std::endl;
                    //     std::cerr << "contact_midtimes[frame][current_contact_idx]: " << contact_midtimes[frame][current_contact_idx] << std::endl;
                    //     throw std::runtime_error("[Reference Generator] contact index issue!");
                    // }
                }
            }
            if (node_foot_pos[frame].size() != nodes_) {
                throw std::runtime_error("[Reference Generator] node_foot_pos size != nodes_");
            }
        }


        // Print for debugging
        // for (const auto& frame : contact_frames_) {
        //     std::cout << frame << ": " << std::endl;
        //     for (int node = 0; node < nodes_; node++) {
        //         std::cout << node_foot_pos[frame][node].transpose() << ", " << contact_schedule.InContact(frame, GetTime(node)) << std::endl;
        //     }
        // }

        // Then do IK on the floating base and leg joints to find a position that fits this step location
        vectorx_t base_config(7);
        vectorx_t q_ik;

        // TODO: May want to do this IK with the floating base when the leg is the most extended, not on the midtime points!
        // TODO: Probably want to do this for both the end of the swing points (most extended) and the mid points of the contacts
        std::vector<std::pair<double, vectorx_t>> times_and_bases;
        std::vector<double> base_times;
        // For all the contact frames
        for (const auto& frame : contact_frames_) {
            // For all the contact midpoint times
            for (const auto& time : contact_midtimes[frame]) {
                // Only check if we haven't used this time
                if (std::find(base_times.begin(), base_times.end(), time) == base_times.end() && time <= end_time_ && time >= 0) {
                    const int node = GetNode(time);

                    // Determine all feet locations at the give time
                    std::vector<vector3_t> foot_pos(contact_frames_.size());
                    for (int j = 0; j < contact_frames_.size(); j++) {
                        foot_pos[j] << node_foot_pos[contact_frames_[j]][node], swing_traj.at(frame)[node];
                    }


                    base_config << GetCommandedConfig(GetNode(time), q_target).head<7>();
                    // std::cout << "Base config: " << base_config.transpose() << ", time: " << time << ", end time: " << end_time_ << std::endl;

                    // IK
                    // TODO: Put back to true!
                    q_ik = model_->InverseKinematics(base_config, foot_pos, contact_frames_, q, false);
                    // q_ik = vectorx_t::Zero(config_size_);

                    // Record base and time
                    times_and_bases.emplace_back(time, q_ik.head<7>());
                    base_times.emplace_back(time);
                    // std::cout << "output base: " << q_ik.head<3>().transpose() << std::endl;
                }
            }
        }

        // Now sort on the times so we can interpolate
        auto time_sort = [](std::pair<double, vectorx_t> p1, std::pair<double, vectorx_t> p2) {
            return p1.first < p2.first;
        };
        std::sort(times_and_bases.begin(), times_and_bases.end(), time_sort);
        // std::cout << "sorted bases:" << std::endl;
        // for (const auto& tb : times_and_bases) {
        //     std::cout << tb.first << std::endl;
        //     // std::cout << tb.second.transpose() << std::endl;
        // }

        // Lastly, run IK on the intermediate nodes with the fixed floating base positions
        q_ref[0] = q;
        std::vector<vector3_t> end_effectors_pos(contact_frames_.size());
        for (int i = 1; i < nodes_; i++) {
            for (int j = 0; j < contact_frames_.size(); j++) {
                const auto& frame = contact_frames_[j];
                end_effectors_pos[j] << node_foot_pos[frame][i].head<2>(), swing_traj.at(frame)[i];
            }

            // TODO: Put back
            base_config << GetBasePositionInterp(GetTime(i), times_and_bases, q_target, q).head<7>();
            // base_config << GetCommandedConfig(i, q_target).head<7>();
            // std::cout << "Base config second time: " << base_config.transpose() << std::endl;
            q_ref[i] = model_->InverseKinematics(base_config, end_effectors_pos, contact_frames_, q_ref[i - 1], false);
        }

        // Return
        for (int node = 0; node < nodes_; node++) {
            // q_ref[node] = q;
            v_ref[node] = v_target[node];
        }
        return {q_ref, v_ref};
    }

    int ReferenceGenerator::GetNode(double time) {
        if (time < 0) {
            throw std::runtime_error("[Reference Generator] time for node is less than 0.");
        }

        if (time > end_time_) {
            throw std::runtime_error("[Reference Generator] time for node is greater than trajectory end time. "
                                     "Time: " + std::to_string(time) + " end time: " + std::to_string(end_time_));
        }

        double t = 0;
        int i = 0;
        while (t <= time) {
            t += dt_[i];
            i++;
        }

        if (i-1 >= nodes_) {
            throw std::runtime_error("[Reference Generator] Node too large!");
        }

        return i-1;
    }

    double ReferenceGenerator::GetTime(int node) {
        double t = 0;
        int i = 0;
        while (i < node) {
            t += dt_[i];
            i++;
        }

        return t;
    }

    vectorx_t ReferenceGenerator::GetCommandedConfig(double time, const SimpleTrajectory& q_target, const SimpleTrajectory& v_target) {
        if (time <= end_time_) {
            double t = 0;
            int i = 0;
            while (t <= time) {
                t += dt_[i];
                i++;
            }

            if (i > 0) {
                i = i - 1;
                t = t - dt_[i];
            }

            double lambda = (time - t) / dt_[i];

            // if (time > 0.78) {
            //     std::cerr << "time: " << time << ", t: " << t << ", i: " << i << ", lambda: " << lambda << std::endl;
            // }

            // Interpolate between the two closest points
            if (lambda > 1.01 || lambda < 0) {
                std::cerr << "time: " << time << ", t: " <<  t << ", lambda: " << lambda << ", dt: " << dt_[i] << std::endl;
                throw std::runtime_error("[Reference Generator] Lambda error!");
            }
            vectorx_t commanded_pos = q_target[i]; //lambda*q_target[i] + (1 - lambda)*q_target[i-1];

            return commanded_pos;
        } else {
            vectorx_t commanded_pos = q_target[nodes_ - 1]; // TODO: Put back! + (time - end_time_)*v_target[nodes_ - 1];
            return commanded_pos;
        }
    }

    vectorx_t ReferenceGenerator::GetCommandedConfig(int node, const SimpleTrajectory &q_target) {
        return q_target[node];
    }

    vectorx_t ReferenceGenerator::GetBasePositionInterp(double time,
        const std::vector<std::pair<double, vectorx_t>>& times_and_bases, const SimpleTrajectory& q_target, const vectorx_t& q_init) {
        int i = 0;
        while (i < times_and_bases.size() && times_and_bases.at(i).first < time) {
            i++;
        }

        if (i == 0) {
            // Interpolate from current base configuration
            double lambda = (time)/(times_and_bases[i].first);
            if (lambda > 1.0 || lambda < 0) {
                std::cerr << "time: " << time << ", lambda: " << lambda << std::endl;
                throw std::runtime_error("[Reference Generator] Lambda error!");
            }
            return lambda*times_and_bases[i].second + (1-lambda)*q_init.head<7>();
        }
        if (i == times_and_bases.size()) {
            // When we are not between any two midpoints, we want to interpolate between the last mid point and the last commanded value
            double lambda = (time - times_and_bases[i-1].first)/(end_time_ - times_and_bases[i-1].first);

            return lambda*GetCommandedConfig(nodes_-1, q_target) + (1-lambda)*times_and_bases[i-1].second;
        }

        double lambda = (time - times_and_bases[i-1].first)/(times_and_bases[i].first - times_and_bases[i-1].first);
        if (lambda > 1.0 || lambda < 0) {
            std::cerr << "time: " << time << ", lambda: " << lambda << std::endl;
            throw std::runtime_error("[Reference Generator] Lambda error!");
        }
        return lambda*times_and_bases.at(i).second + (1-lambda)*times_and_bases.at(i-1).second;
    }
}
