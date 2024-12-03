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

    // TODO: WIP. Not working!
    // TODO: Incorporate the angular velocities
    std::pair<SimpleTrajectory, SimpleTrajectory> ReferenceGenerator::GenerateReference(const vectorx_t& q,
                                                                                        const vectorx_t& v,
                                                                                        const vector3_t& commanded_vel,
                                                                                        const std::map<std::string, std::vector<double>>& swing_traj,
                                                                                        const std::vector<double>& hip_offsets,
                                                                                        const std::map<std::string, std::vector<int>>& in_contact,
                                                                                        const ContactSchedule& contact_schedule) {
        if (hip_offsets.size() != 2*contact_frames_.size()) {
            throw std::runtime_error("Hip offsets size != 2*contact_frames_.size()");
        }

        SimpleTrajectory q_ref(config_size_, nodes_);
        SimpleTrajectory v_ref(vel_size_, nodes_);

        std::map<std::string, std::vector<double>> contact_midtimes;
        std::map<std::string, std::vector<int>> contact_midtimes_nodes;
        // std::vector<int> contact_nodes_all;
        std::vector<vector3_t> contact_base_positions;
        std::vector<vector3_t> ik_base_positions(nodes_);

        // First determine the nodes (and times) associated with the midpoints of the contacts
        const auto contact_map = contact_schedule.GetScheduleMap();
        for (const auto& [frame, swings] : contact_map) {
            contact_midtimes.insert({frame, {}});
            contact_midtimes_nodes.insert({frame, {}});

            for (int i = 0; i < swings.size() + 1; i++) {
                if (swings.size() > 0) {
                    if (i == 0) {
                        // Handle the contact midpoint before the first swing
                        contact_midtimes[frame].emplace_back(std::min(
                            std::max(0.0, swings[0].first - 0.15), end_time_ - 0.1)); // TODO: Remove the -0.01
                    } else if (i == swings.size()) {
                        // Handle the contact midpoint after the last swing
                        contact_midtimes[frame].emplace_back(swings[i - 1].second + 0.15);
                    } else {
                        contact_midtimes[frame].emplace_back((swings[i+1].first + swings[i].second)/2.0);
                    }
                } else {
                    contact_midtimes[frame].emplace_back(end_time_/2.0);
                }
            }

            if (contact_midtimes[frame].size() <= 0) {
                throw std::runtime_error("[Reference Generator] contact midtimes not filled correctly!");
            }

            for (int i = 0; i < contact_midtimes[frame].size(); i++) {
                if (contact_midtimes[frame][i] <= end_time_ && contact_midtimes[frame][i] >= 0) {
                    contact_midtimes_nodes[frame].emplace_back(GetNode(contact_midtimes[frame][i]));
                    if (contact_midtimes_nodes[frame][contact_midtimes_nodes[frame].size()-1] >= nodes_) {
                        throw std::runtime_error("[Reference Generator] Node is too large!");
                    }
                } else {
                    std::cerr << "Contact mid time: " << contact_midtimes[frame][i] << " end time: " << end_time_ << std::endl;
                }
            }

            if (contact_midtimes_nodes[frame].size() <= 0) {
                throw std::runtime_error("[Reference Generator] contact midtimes nodes not filled correctly!");
            }
        }

        // for (const auto& frame : contact_frames_) {
        //     std::cout << frame << std::endl;
        //     for (const auto& node : contact_midtimes_nodes[frame]) {
        //         std::cout << node << ", ";
        //     }
        //     std::cout << std::endl;
        // }


        // for (const auto& [frame, times] : contact_midtimes) {
        //     std::cout << frame << " mid contact times:" << std::endl;
        //     for (const auto& t : times) {
        //         std::cout << t << std::endl;
        //     }
        // }

        // Next, determine where the foot should be using the desired location and footstep constraints
        // Determine the "commanded" positions
        std::vector<vector3_t> commanded_pos(nodes_);
        commanded_pos[0] = q.head<3>();
        for (int i = 1; i < commanded_pos.size(); i++) {
            commanded_pos[i] = commanded_pos[i - 1] + dt_[i]*commanded_vel;
            // std::cout << "commanded pos: " << commanded_pos[i] << std::endl;
        }

        // Get the contact location based on the midpoints
        // Interpolate between these during the swing phases
        // Constant position during contacts
        // Add hip offsets
        std::map<std::string, std::vector<vector3_t>> foot_pos;
        model_->FirstOrderFK(q);
        for (int j = 0; j < contact_frames_.size(); j++) {
            std::string& frame = contact_frames_[j];
            foot_pos.insert({frame, {}});
            foot_pos[frame].resize(nodes_);

            // Current foot position
            foot_pos[frame][0] = model_->GetFrameState(frame).placement.translation();

            vector3_t hip_offset;
            hip_offset << hip_offsets[2*j], hip_offsets[2*j + 1], 0;    // TODO: Change height

            // Contact locations
            for (const auto& node : contact_midtimes_nodes[frame]) {
                if (node >= commanded_pos.size()) {
                    throw std::runtime_error("[Reference generator] contact midpoint node calculation error.");
                }
                foot_pos[frame][node] = hip_offset + commanded_pos[node];
            }

            // Filling in the contact locations
            int contact_idx = 0;
            for (int node = 0; node < nodes_; node++) {
                if (in_contact.at(frame)[node]) {
                    std::cerr << frame << std::endl;
                    std::cerr << "node: " << node << std::endl;
                    std::cerr << "contact node size: " << contact_midtimes_nodes[frame].size() << std::endl;
                    std::cerr << "contact node: " << contact_midtimes_nodes[frame][contact_idx] << std::endl;
                    foot_pos[frame][node] = foot_pos[frame][contact_midtimes_nodes[frame][contact_idx]];
                } else if ((node > 0 && in_contact.at(frame)[node - 1] && !in_contact.at(frame)[node]) || node == 0) {
                    contact_idx++;
                }
            }

            vector2_t next_foothold, prev_foothold;
            int swing_start = 0;
            int swing_end = 0;
            for (int node = 0; node < nodes_; node++) {
                // Is this node the start of a swing?
                if ((node == 0 && !in_contact.at(frame)[node]) ||
                    (node > 0 && (!in_contact.at(frame)[node] && in_contact.at(frame)[node-1]))) {
                    swing_start = node;
                }

                // If it is, determine the end of the swing
                if (swing_start == node) {
                    swing_end = swing_start;
                    while (swing_end < nodes_ && !in_contact.at(frame).at(swing_end)) {
                        swing_end++;
                    }
                }

                // Dealing with running out of nodes
                if (swing_end == nodes_) {
                    // swing_end += 5; // TODO: Deal with this better, for now just extending it a bit
                    // TODO: Consider changing how this next foothold is computed
                    next_foothold = prev_foothold + dt_[1]*(swing_end - swing_start)*commanded_vel.head<2>();
                } else {
                    next_foothold = foot_pos[frame].at(swing_end).head<2>();
                }

                // If in a swing phase (between swing_start and swing_end) then interpolate on location
                if (!in_contact.at(frame)[node]) {
                    vector2_t swing_location = prev_foothold + (next_foothold - prev_foothold)*(static_cast<double>(node - swing_start)/static_cast<double>(swing_end - swing_start));
                    foot_pos[frame][node] << swing_location(0), swing_location(1), swing_traj.at(frame)[node];
                }
            }

            // std::cout << frame << " foot_pos[" << 0 << "]: " << foot_pos[frame][0].transpose() << std::endl;
        }

        // Print current foot positions for debugging
        // model_->FirstOrderFK(q);
        // for (const auto& frame : contact_frames_) {
        //     std::cout << frame << ": " << model_->GetFrameState(frame).placement.translation().transpose() << std::endl;
        // }

        // Then do IK on the floating base and leg joints to find a position that fits this step location
        vectorx_t base_config(7);
        std::vector<int> contact_nodes_all;
        std::vector<std::vector<std::string>> frames(nodes_);
        std::vector<std::vector<vector3_t>> foot_pos_vec(nodes_);
        // for (const auto& frame : contact_frames_) {
        for (int node = 0; node < nodes_; node++) {
            for (const auto& frame : contact_frames_) {
                if (std::find(contact_midtimes_nodes[frame].begin(), contact_midtimes_nodes[frame].end(), node)
                    != contact_midtimes_nodes[frame].end()) {
                    if (contact_nodes_all.empty() || std::find(contact_nodes_all.begin(), contact_nodes_all.end(), node)
                        == contact_nodes_all.end()) {
                        contact_nodes_all.emplace_back(node);
                        // std::cout << "inserting node: " << node << std::endl;
                    }
                    frames[node].emplace_back(frame);
                    foot_pos_vec[node].emplace_back(foot_pos[frame][node]);
                    // std::cout << "Contact at: frame " << frame << ", node: " << node << std::endl;
                }
            }
        }

        for (const auto& node : contact_nodes_all) {
            base_config << commanded_pos[node], q.segment<4>(3);
            q_ref[node] = model_->InverseKinematics(base_config, foot_pos_vec[node], frames[node], q, true);
            contact_base_positions.emplace_back(q_ref[node].head<3>());
            // std::cout << "Contact base position: " << contact_base_positions[contact_base_positions.size() - 1] << std::endl;

            // Print positions for debugging
            // std::cout << "After IK:" << std::endl;
            // model_->FirstOrderFK(q_ref[node]);
            // for (const auto& frame : contact_frames_) {
            //     std::cout << frame << ": " << model_->GetFrameState(frame).placement.translation().transpose() << std::endl;
            // }
            // std::cout << "desired:" << std::endl;
            // for (const auto& pos: foot_pos_vec[node]) {
            //     std::cout << pos.transpose() << std::endl;
            // }
            // std::cout << std::endl;
        }

        //     for (int i = 0; i < contact_midtimes_nodes[frame].size(); i++) {
        //         int node = contact_midtimes_nodes[frame][i];
        //         contact_nodes_all.emplace_back(node);
        //         base_config << commanded_pos[node], q.segment<4>(3);
        //         foot_pos_vec[0] = foot_pos[node];
        //         q_ref[node] = model_->InverseKinematics(base_config, foot_pos_vec, frames, q, true);
        //         contact_base_positions.emplace_back(q_ref[node].head<3>());
        //     }
        // }

        // std::cout << "contact_nodes_all.size(): " << contact_nodes_all.size() << std::endl;

        // std::cout << "q: " << q.transpose() << std::endl;

        //////
        // Then interpolate between floating base positions to get a floating base trajectory
        ik_base_positions[0] = q.head<3>();
        int contact_idx = 0;
        for (int i = 0; i < nodes_; i++) {
            if (contact_idx == 0) {
                int contact_node = contact_nodes_all[contact_idx];
                double lambda = (static_cast<float>(i*contact_node))/(contact_node*contact_node);
                // std::cout << "lambda: " << lambda << ", contact idx: " << contact_idx << " contact node: " << contact_node << std::endl;
                // TODO: Check
                ik_base_positions[i] = lambda*contact_base_positions[contact_idx] + (1 - lambda)*ik_base_positions[0];
            } else {
                float prev_contact_node = contact_nodes_all[contact_idx - 1];
                float next_contact_node = contact_nodes_all[contact_idx];
                float contact_node_diff = (next_contact_node - prev_contact_node);
                double lambda = ((i - prev_contact_node)*contact_node_diff)/(contact_node_diff*contact_node_diff);
                // TODO: Check
                // std::cout << "lambda: " << lambda << ", contact idx: " << contact_idx << " next_contact_node: " << next_contact_node << std::endl;
                ik_base_positions[i] = lambda*contact_base_positions[contact_idx] + (1 - lambda)*contact_base_positions[contact_idx - 1];
            }

            // std::cout << "ik_base_positions: " << ik_base_positions[i].transpose() << std::endl;

            if (i >= contact_nodes_all[contact_idx]) {
                contact_idx++;
                // std::cout << "contact idx: " << contact_idx << std::endl;
            }
        }

        // TODO: Debug
        // Lastly, run IK on the intermediate nodes with the fixed floating base positions
        q_ref[0] = q;
        std::vector<vector3_t> end_effectors_pos(contact_frames_.size());
        for (int i = 1; i < nodes_; i++) {
            for (int j = 0; j < contact_frames_.size(); j++) {
                const auto& frame = contact_frames_[j];
                end_effectors_pos[j] << foot_pos[frame][i].head<2>(), swing_traj.at(frame)[i];
            }
            // TODO: Put back
            // base_config << ik_base_positions[i], q.segment<4>(3);
            base_config << commanded_pos[i], q.segment<4>(3);
            q_ref[i] = model_->InverseKinematics(base_config, end_effectors_pos, contact_frames_, q_ref[i - 1], false);
            if (q_ref[i].norm() > 100) {
                throw std::runtime_error("q_ref[i] > 100");
            }
        }

        // Return
        // for (int node = 0; node < nodes_; node++) {
        //     q_ref[node] = q;
        //     v_ref[node] = vectorx_t::Zero(vel_size_);
        // }
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

}