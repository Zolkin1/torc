//
// Created by zolkin on 12/2/24.
//

#include "reference_generator.h"

#include <torc_timer.h>

namespace torc::mpc {
    ReferenceGenerator::ReferenceGenerator(int nodes, const std::vector<std::string>& contact_frames,
                                           const std::vector<double>& dt, const models::FullOrderRigidBody& model,
                                           double polytope_delta)
    : nodes_(nodes), config_size_(model.GetConfigDim()), vel_size_(model.GetVelDim()), qp_(2,0,2),
        dt_(dt), contact_frames_(contact_frames), model_(model) {
        end_time_ = 0;
        for (const auto d : dt_) {
            end_time_ += d;
        }

        polytope_delta_ = polytope_delta;
    }

    // TODO: Clean up!
    // TODO: Consider modulating the commanded velocity and position dependent on the polytopes.
    //  e.g. if we are asked to walk off a polytope and no other polytope is selected then we should modulate the position
    //  and velocity to keep the body in the polytope, which will make the optimization easier. Just need to be careful
    //  that this is done correctly. Can also increase velocity/position if making ot over a big gap.
    std::pair<SimpleTrajectory, SimpleTrajectory> ReferenceGenerator::GenerateReference(const vectorx_t& q, const vectorx_t& v,
        SimpleTrajectory q_target, // TODO: Should I use a reference instead?
        SimpleTrajectory v_target,
        const std::map<std::string, std::vector<double>>& swing_traj,
        const std::vector<double>& hip_offsets,
        const ContactSchedule& contact_schedule,
        std::map<std::string, std::vector<vector3_t>>& des_foot_pos) {
        // std::cerr << "In reference generator!" << std::endl;

        if (hip_offsets.size() != 2*contact_frames_.size()) {
            throw std::runtime_error("Hip offsets size != 2*contact_frames_.size()");
        }

        SimpleTrajectory q_ref(config_size_, nodes_);
        SimpleTrajectory v_ref(vel_size_, nodes_);

        std::map<std::string, std::vector<double>> contact_midtimes;
        std::vector<vector3_t> contact_base_positions;
        std::vector<vector3_t> ik_base_positions(nodes_);

        // std::cerr << "About to start contact mid times!" << std::endl;
        // -------------------------------------------------- //
        // -------------------------------------------------- //
        // Compute contact mid point times
        // -------------------------------------------------- //
        // -------------------------------------------------- //
        const auto& contact_map = contact_schedule.GetScheduleMap();
        for (const auto& [frame, swings] : contact_map) {
            // std::cerr << "Frame: " << frame << std::endl;
            // std::cerr << "swings[0]: " << swings[0].second << " first: " << swings[0].first << std::endl;
            contact_midtimes.insert({frame, {}});

            double swing_time = swings[0].second - swings[0].first;
            if (!swings.empty()) {
                // Handle the contact midpoint before the first swing
                contact_midtimes[frame].emplace_back(swings[0].first - swing_time/2.0);
                // contact_midtimes[frame].emplace_back(std::max(0.0, swings[0].first - 0.15));
                for (int i = 0; i < swings.size() - 1; i++) {
                    contact_midtimes[frame].emplace_back((swings[i+1].first + swings[i].second)/2.0);
                }
                // Handle the contact midpoint after the last swing
                contact_midtimes[frame].emplace_back(swings[swings.size() - 1].second + swing_time/2.0);
            } else {
                contact_midtimes[frame].emplace_back(end_time_/2.0);
            }

            // std::cout << "frame: " << frame << std::endl;
            // for (const auto& time : contact_midtimes[frame]) {
            //     std::cout << time << " ";
            // }
            // std::cout << std::endl;
            // std::cerr << "Starting debug checks!" << std::endl;

            // DEBUG CHECK
            if (contact_schedule.GetPolytopes(frame).size() != contact_midtimes[frame].size()) {
                throw std::runtime_error("[Reference generator] Polytopes size != contact_midtimes.size()");
            }
            // DEBUG CHECK
            if (contact_midtimes[frame].size() != contact_schedule.GetNumContacts(frame)) {
                std::cerr << "frame: " << frame << std::endl;
                std::cerr << "contact_midtimes size: " << contact_midtimes[frame].size() << std::endl;
                std::cerr << "num contacts: " << contact_schedule.GetNumContacts(frame) << std::endl;
                throw std::runtime_error("[Reference generator] NumContacts != contact_midtimes.size()");
            }
        }

        std::map<std::string, int> polytope_idx_offset;     // Account for the deleted times when accessing the polytopes
        // Remove all negative time contacts
        for (auto& [frame, midtimes] : contact_midtimes) {
            polytope_idx_offset.insert({frame, 0});
            for (int i = 0; i < midtimes.size(); i++) {
                if (midtimes[i] < 0) {
                    midtimes.erase(midtimes.begin() + i);
                    i--;
                    polytope_idx_offset[frame]++;
                }
            }
        }

        // std::cerr << "contact midtimes determined!" << std::endl;

        // -------------------------------------------------- //
        // -------------------------------------------------- //
        // Compute the x-y position of the feet at every node
        // -------------------------------------------------- //
        // -------------------------------------------------- //
        // Holds the position of the feet at the contact midpoints for each frame
        std::map<std::string, std::vector<vector2_t>> contact_foot_pos;
        std::map<double, vector2_t> base_pos;
        des_foot_pos.clear();
        // std::map<std::string, std::vector<vector2_t>> des_foot_pos;
        model_.FirstOrderFK(q);
        const auto& schedule = contact_schedule.GetScheduleMap();
        for (int j = 0; j < contact_frames_.size(); j++) {
            std::string& frame = contact_frames_[j];

            contact_foot_pos.insert({frame, {}});
            des_foot_pos.insert({frame, {}});

            vector3_t hip_offset;
            hip_offset << hip_offsets[2*j], hip_offsets[2*j + 1], 0;

            // Determine all contact locations for the given frame
            for (int i = 0; i < contact_midtimes[frame].size(); i++) {
                // Get the contact location based on the midpoints and hip offsets
                // Get rotation matrix for the hip offsets

                if (contact_midtimes[frame][i] >= 0) {
                    double time = contact_midtimes[frame][i];
                    vectorx_t q_command = GetCommandedConfig(time, q_target, v_target);
                    const quat_t quat(q_command.segment<4>(3));
                    const matrix3_t R = quat.toRotationMatrix();    // TODO: Should this be transposed?

                    // TODO: Verify this is correct when tilted!
                    // contact_foot_pos[frame].emplace_back((R*hip_offset).head<2>()
                    //     + q_command.head<2>());
                    contact_foot_pos[frame].emplace_back(R.topLeftCorner<2,2>()*hip_offset.head<2>().head<2>() + q_command.head<2>());

                    // TODO: Put back!
                    // TODO: Need to use the MPC desired trajectory I think rather than the commanded traj
                    //  As it is right now, I get oscillations but the MPC is commanding the velocities seen as error
                    // Raibert Heuristic
                    if (i == 0) {
                        // For now just in the plane
                        constexpr double g = 9.81;
                        const double hnom = q_target[0][2];
                        contact_foot_pos[frame].back() = contact_foot_pos[frame].back() + (R.topLeftCorner<2,2>()*std::sqrt(hnom/g)*(v.head<2>() - v_target[0].head<2>())).head<2>();
                    }

                    // std::cout << "[RG] time: " << time << ", b: " <<
                    //     contact_schedule.GetPolytopes(frame).at(i + polytope_idx_offset[frame]).b_.transpose() << std::endl;

                    int contact_idx_temp = contact_schedule.GetContactIndex(frame, time);
                    // std::cout << "contact_idx_temp: " << contact_idx_temp << ", polytope idx: " << i + polytope_idx_offset[frame] << std::endl;
                    if (contact_idx_temp != i + polytope_idx_offset[frame]) {
                        throw std::runtime_error("[Reference generator] contact_idx_temp != contact_idx_temp");
                    }

                    // Project onto the polytope
                    bool projected = ProjectOnPolytope(contact_foot_pos[frame].back(),
                        contact_schedule.GetPolytopes(frame).at(i + polytope_idx_offset[frame]));
                    if (time < end_time_ && !base_pos.contains(time)) {
                        // This was removed because otherwise the foot would never get a chance to move to the next stone
                        // if (projected) {
                        //     base_pos.insert({time, contact_foot_pos[frame].back() - R.topLeftCorner<2,2>()*hip_offset});
                        // } else {
                            base_pos.insert({time, q_target[GetNode(time)].head<2>()});
                        // }
                    }
                } else {
                    throw std::runtime_error("Negative time!");
                }
            }

            // std::cerr << "contact locations determined!" << std::endl;

            if (contact_foot_pos[frame].size() != contact_midtimes[frame].size()) {
                throw std::runtime_error("[Reference generator] FootPos size != contact_midtimes.size()");
            }

            int current_contact_idx = 0;
            for (int node = 0; node < nodes_; node++) {
                // Get the contact positions in front and behind the current time
                double time = GetTime(node);

                // Check if we are in swing
                if (contact_schedule.InSwing(frame, time)) {
                    if (node > 0 && contact_schedule.InContact(frame, GetTime(node - 1))) {
                        // std::cerr << "[" << frame << "] " << "updating contact idx at node " << node << "(time " << time << ")" << std::endl;
                        current_contact_idx++;
                    }

                    if (current_contact_idx < 0) {
                        throw std::runtime_error("[Reference Generator] contact index issue! "
                                                 "Contact idx: " + std::to_string(current_contact_idx) + " time: " + std::to_string(time));
                    }

                    vector2_t swing_intermediate_pos;
                    double swing_duration = contact_schedule.GetSwingDuration(frame, time);
                    if (current_contact_idx > 0) {
                        double swing_start = contact_schedule.GetSwingStartTime(frame, time);
                        double lambda = (time - swing_start)/(swing_duration);

                        if (lambda > 1 || lambda < 0) {
                            throw std::runtime_error("[Reference Generator] lambda issue!");
                        }

                        // DEBUG CHECK
                        if (contact_foot_pos[frame].size() <= current_contact_idx) {
                            std::cerr << "current_contact_idx: " << current_contact_idx << std::endl;
                            std::cerr << "contact_foot_pos.size(): " << contact_foot_pos[frame].size() << std::endl;
                            std::cerr << "num contacts: " << contact_schedule.GetNumContacts(frame) << std::endl;
                            std::cerr << "node: " << node << " frame: " << frame << " time: " << time << std::endl;
                            std::cerr << "contact mid times: " << std::endl;
                            for (int k = 0; k < contact_midtimes[frame].size(); k++) {
                                std::cerr << contact_midtimes[frame][k] << ", ";
                            }
                            std::cerr << std::endl;
                            throw std::runtime_error("[Reference Generator] contact_foot_pos size issue!");
                        }

                        swing_intermediate_pos = lambda*contact_foot_pos[frame].at(current_contact_idx)
                            + (1-lambda)*contact_foot_pos[frame][current_contact_idx-1];
                        // if (node < nodes_ - 1 && contact_schedule.InContact(frame, GetTime(node + 1))) {
                        //     std::cerr << "frame " << frame << " node " << node << std::endl;
                        //     std::cerr << "lambda: " << lambda << std::endl;
                        //     std::cerr << "swing pos: " << swing_intermediate_pos.transpose() << std::endl;
                        //     std::cerr << "next contact: " << contact_foot_pos[frame][current_contact_idx].transpose() << std::endl;
                        // }
                    } else if (current_contact_idx == 0) {
                        double swing_start = contact_schedule.GetSwingStartTime(frame, time);
                        double lambda = (time - swing_start)/(swing_duration);

                        if (lambda > 1 || lambda < 0) {
                            std::cerr << "lambda: " << lambda << std::endl;
                            std::cerr << "swing_duration: " << swing_duration << std::endl;
                            std::cerr << "swing_start: " << swing_start << std::endl;
                            throw std::runtime_error("[Reference Generator] lambda issue!");
                        }

                        swing_intermediate_pos = lambda*contact_foot_pos[frame][current_contact_idx]
                            + (1-lambda)*model_.GetFrameState(frame).placement.translation().head<2>();

                        // if (node < nodes_ - 1 && contact_schedule.InContact(frame, GetTime(node + 1))) {
                        //     std::cerr << "frame " << frame << " node " << node << std::endl;
                        //     std::cerr << "time: " << time << " swing start: " << swing_start << " swing duration: " << swing_duration << std::endl;
                        //     std::cerr << "lambda: " << lambda << std::endl;
                        //     std::cerr << "swing pos: " << swing_intermediate_pos.transpose() << std::endl;
                        //     std::cerr << "next contact: " << contact_foot_pos[frame][current_contact_idx].transpose() << std::endl;
                        // }
                    }

                    vector3_t foot_pos;
                    foot_pos << swing_intermediate_pos, swing_traj.at(frame)[node];
                    des_foot_pos[frame].emplace_back(foot_pos);
                } else {
                    if (!contact_schedule.InContact(frame, GetTime(node))) {
                        throw std::runtime_error("[Reference Generator] should be in contact!");
                    }
                    // DEBUG CHECK
                    if (contact_foot_pos[frame].size() <= current_contact_idx) {
                        std::cerr << "current_contact_idx: " << current_contact_idx << std::endl;
                        std::cerr << "contact_foot_pos.size(): " << contact_foot_pos[frame].size() << std::endl;
                        std::cerr << "num contacts: " << contact_schedule.GetNumContacts(frame) << std::endl;
                        std::cerr << "node: " << node << " frame: " << frame << " time: " << time << std::endl;
                        std::cerr << "contact mid times: " << std::endl;
                        for (int k = 0; k < contact_midtimes[frame].size(); k++) {
                            std::cerr << contact_midtimes[frame][k] << ", ";
                        }
                        std::cerr << std::endl;
                        throw std::runtime_error("[Reference Generator] contact_foot_pos size issue!");
                    }

                    if (node == 0) {
                        model_.FirstOrderFK(q);
                        double next_swing_duration = contact_schedule.GetNextSwingDuration(frame, time);
                        // If we deleted the contact mid time but it is still in use then we need to make a position for it
                        if (contact_midtimes[frame].at(current_contact_idx) > time + next_swing_duration) {
                            // Insert a contact_foot_pos at the front with the current position
                            contact_foot_pos[frame].insert(contact_foot_pos[frame].begin(),
                                model_.GetFrameState(frame).placement.translation().head<2>());
                        } else {
                            contact_foot_pos[frame].at(current_contact_idx) = model_.GetFrameState(frame).placement.translation().head<2>();
                        }
                    }

                    vector3_t foot_pos;
                    foot_pos << contact_foot_pos[frame].at(current_contact_idx), swing_traj.at(frame)[node];
                    des_foot_pos[frame].emplace_back(foot_pos);
                }
            }
            if (des_foot_pos[frame].size() != nodes_) {
                throw std::runtime_error("[Reference Generator] node_foot_pos size != nodes_");
            }
        }

        // std::cerr << "swing locations determined!" << std::endl;


        vectorx_t base_config(7);
        vectorx_t q_ik;

        // DEBUG CHECK
        // for (const auto& [frame, positions] : node_foot_pos) {
        //     for (int node = 0; node < nodes_ - 1; node++) {
        //         if (contact_schedule.InContact(frame, GetTime(node)) && contact_schedule.InContact(frame, GetTime(node + 1))) {
        //             if (node_foot_pos[frame][node] != node_foot_pos[frame][node + 1]) {
        //                 std::cerr << "Frame: " << frame << " Node 1: " << node << " Node 2: " << node + 1 << std::endl;
        //                 std::cerr << "Position 1: " << positions[node].transpose() << std::endl;
        //                 std::cerr << "Position 2: " << positions[node + 1].transpose() << std::endl;
        //                 throw std::runtime_error("[Reference Generator] node_foot_pos size issue!");
        //             }
        //         }
        //
        //         if (contact_schedule.InContact(frame, GetTime(node + 1)) && contact_schedule.InSwing(frame, GetTime(node))) {
        //             if ((node_foot_pos[frame][node] - node_foot_pos[frame][node + 1]).norm() > 0.095) {
        //                 std::cerr << "Frame: " << frame << " Node 1: " << node << " Node 2: " << node + 1 << std::endl;
        //                 std::cerr << "Position 1: " << positions[node].transpose() << std::endl;
        //                 std::cerr << "Position 2: " << positions[node + 1].transpose() << std::endl;
        //                 throw std::runtime_error("[Reference Generator] swing traj not ending near contact position!");
        //             }
        //         }
        //     }
        // }

        // -------------------------------------------------- //
        // -------------------------------------------------- //
        // Now adjust the configuration targets by interpolating through the base positions
        // -------------------------------------------------- //
        // -------------------------------------------------- //
        const quat_t quat(q_target[nodes_-1].segment<4>(3));
        const matrix3_t R_end = quat.toRotationMatrix();
        const vector2_t v_end_command = R_end.topLeftCorner<2,2>()*v_target[nodes_-1].head<2>();
        for (int node = 0; node < nodes_; node++) {
            q_target[node].head<2>() = InterpolateBasePositions(node, base_pos, q.head<2>(), v_end_command);
        }

        // std::cerr << "base interp completed!" << std::endl;

        // -------------------------------------------------- //
        // -------------------------------------------------- //
        // Adjust base position to avoid IK issues (for now skipping)
        // -------------------------------------------------- //
        // -------------------------------------------------- //
        // // Then do IK on the floating base and leg joints to find a position that fits this step location
        //
        // // TODO: May want to do this IK with the floating base when the leg is the most extended, not on the midtime points!
        // // TODO: Probably want to do this for both the end of the swing points (most extended) and the mid points of the contacts
        // std::vector<std::pair<double, vectorx_t>> times_and_bases;
        // std::vector<double> base_times;
        // // For all the contact frames
        // for (const auto& frame : contact_frames_) {
        //     // For all the contact midpoint times
        //     for (const auto& time : contact_midtimes[frame]) {
        //         // Only check if we haven't used this time
        //         if (std::find(base_times.begin(), base_times.end(), time) == base_times.end() && time <= end_time_ && time >= 0) {
        //             const int node = GetNode(time);
        //
        //             // Determine all feet locations at the give time
        //             std::vector<vector3_t> foot_pos(contact_frames_.size());
        //             for (int j = 0; j < contact_frames_.size(); j++) {
        //                 foot_pos[j] << node_foot_pos[contact_frames_[j]][node], swing_traj.at(frame)[node];
        //             }
        //
        //
        //             base_config << GetCommandedConfig(GetNode(time), q_target).head<7>();
        //             // std::cout << "Base config: " << base_config.transpose() << ", time: " << time << ", end time: " << end_time_ << std::endl;
        //
        //             // IK
        //             // TODO: Put back to true!
        //             q_ik = model_->InverseKinematics(base_config, foot_pos, contact_frames_, q, false);
        //
        //             // Record base and time
        //             times_and_bases.emplace_back(time, q_ik.head<7>());
        //             base_times.emplace_back(time);
        //         }
        //     }
        // }
        //
        // // Now sort on the times so we can interpolate
        // auto time_sort = [](std::pair<double, vectorx_t> p1, std::pair<double, vectorx_t> p2) {
        //     return p1.first < p2.first;
        // };
        // std::sort(times_and_bases.begin(), times_and_bases.end(), time_sort);

        // -------------------------------------------------- //
        // -------------------------------------------------- //
        // Run IK on all the nodes with the fixed base positions
        // -------------------------------------------------- //
        // -------------------------------------------------- //
        // Lastly, run IK on the intermediate nodes with the fixed floating base positions
        q_ref[0] = q;
        std::vector<vector3_t> end_effectors_pos(contact_frames_.size());
        for (int i = 1; i < nodes_; i++) {
            for (int j = 0; j < contact_frames_.size(); j++) {
                const auto& frame = contact_frames_[j];
                end_effectors_pos[j] << des_foot_pos[frame][i].head<2>(), swing_traj.at(frame)[i];
                // std::cout << "j: " << j << ", ee pos: " << end_effectors_pos[j].transpose() << std::endl;
            }

            // TODO: Put back
            // base_config << GetBasePositionInterp(GetTime(i), times_and_bases, q_target, q).head<7>();
            base_config << GetCommandedConfig(i, q_target).head<7>();
            // TODO: Put back if I want it
            // q_ref[i] = model_.InverseKinematics(base_config, end_effectors_pos, contact_frames_, q_ref[i - 1], false);
            // std::cout << "i: " << i << ", qref: " << q_ref[i].transpose() << std::endl;
        }
        // std::cerr << "IK completed!" << std::endl;

        // -------------------------------------------------- //
        // -------------------------------------------------- //
        // Assign velocity targets
        // -------------------------------------------------- //
        // -------------------------------------------------- //
        for (int node = 0; node < nodes_; node++) {
            // TODO: Should I keep the 0?
            v_ref[node] = 0*v_target[node];
            v_ref[node].head<6>() = v_target[node].head<6>();
        }

        // -------------------------------------------------- //
        // -------------------------------------------------- //
        // Return
        // -------------------------------------------------- //
        // -------------------------------------------------- //
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
            return lambda*times_and_bases[i].second + (1-lambda)*q_init.head<7>();
        }
        if (i == times_and_bases.size()) {
            // When we are not between any two midpoints, we want to interpolate between the last mid point and the last commanded value
            double lambda = (time - times_and_bases[i-1].first)/(end_time_ - times_and_bases[i-1].first);

            return lambda*GetCommandedConfig(nodes_-1, q_target) + (1-lambda)*times_and_bases[i-1].second;
        }

        double lambda = (time - times_and_bases[i-1].first)/(times_and_bases[i].first - times_and_bases[i-1].first);
        return lambda*times_and_bases.at(i).second + (1-lambda)*times_and_bases.at(i-1).second;
    }

    bool ReferenceGenerator::ProjectOnPolytope(vector2_t &foot_position, ContactInfo &polytope) {
        // std::cerr << "A:\n" << polytope.A_ << std::endl;
        // std::cerr << "b:\n" << polytope.b_ << std::endl;

        if (polytope.A_.size() == 0) {
            throw std::runtime_error("[Reference Generator] polytope A is empty!");
        }

        if (polytope.b_.size() == 0) {
            throw std::runtime_error("[Reference Generator] polytope b is empty!");
        }

        vector4_t polytope_margin;
        polytope_margin << 1, 1, -1, -1;
        polytope_margin *= polytope_delta_;
        vector4_t b_modified = polytope.b_ - polytope_margin;

        vector2_t c = polytope.A_*foot_position;
        bool in_polytope = true;
        for (int i = 0; i < polytope.b_.size(); i++) {
            if (i < c.size()) {
                if (c(i) > b_modified(i)) {
                    in_polytope = false;
                }
            } else {
                if (c(i - c.size()) < b_modified(i)) {
                    in_polytope = false;
                }
            }
        }

        if (!in_polytope) {

            matrix2_t H = 2*matrix2_t::Identity();
            vector2_t g = -2*foot_position;

            matrixx_t Aeq(0,0);
            vectorx_t beq(0);

            vector2_t lb, ub;
            lb << std::min(b_modified(0), b_modified(2)), std::min(b_modified(1), b_modified(3));
            ub << std::max(b_modified(0), b_modified(2)), std::max(b_modified(1), b_modified(3));

            torc::utils::TORCTimer qp_timer;
            qp_timer.Tic();

            qp_.init(H, g, Aeq, beq, polytope.A_, lb, ub);

            qp_.solve();

            qp_timer.Toc();

            if (std::abs(qp_.results.info.objValue - ((foot_position - qp_.results.x).squaredNorm() - foot_position.squaredNorm())) > 1e-4) {
                std::cerr << "got: " << qp_.results.info.objValue << std::endl;
                std::cerr << "expected: " << (foot_position - qp_.results.x).squaredNorm() - foot_position.squaredNorm() << std::endl;
                throw std::runtime_error("[Reference Generator] qp not formed correctly!");
            }

            foot_position = qp_.results.x;

            // std::cout << "projection run time: " << qp_timer.Duration<std::chrono::microseconds>().count()/1000.0 << " ms" << std::endl;
            // std::cout << "foot_position: " << foot_position << std::endl;
            return true;
        } else {
            return false;
        }
    }

    vector2_t ReferenceGenerator::InterpolateBasePositions(int node, const std::map<double, vector2_t> &base_pos,
        const vector2_t &current_pos, const vector2_t& end_vel_command) {
        double interp_time = GetTime(node);

        const auto next_it = base_pos.upper_bound(interp_time);
        auto past_it = next_it;
        --past_it;

        if (next_it != base_pos.end() && next_it != base_pos.begin()) {
            double last_base_pos_time = past_it->first;
            double next_base_pos_time = next_it->first;

            double lambda = (interp_time - last_base_pos_time)/(next_base_pos_time - last_base_pos_time);
            if (lambda < 0 || lambda > 1) {
                throw std::runtime_error("1");
            }
            return (1-lambda)*past_it->second + lambda*next_it->second;

        } else if (next_it == base_pos.begin()) {
            double last_base_pos_time = 0;
            double next_base_pos_time = next_it->first;

            double lambda = (interp_time - last_base_pos_time)/(next_base_pos_time - last_base_pos_time);
            if (lambda < 0 || lambda > 1) {
                throw std::runtime_error("2");
            }
            return (1-lambda)*current_pos + lambda*next_it->second;

        } else {
            double last_base_pos_time = past_it->first;
            double next_base_pos_time = end_time_;

            double lambda = (interp_time - last_base_pos_time)/(next_base_pos_time - last_base_pos_time);
            if (lambda < 0 || lambda > 1) {
                throw std::runtime_error("3");
            }
            // TODO: Rotate the vel command
            return (1-lambda)*past_it->second + lambda*(past_it->second + end_vel_command*(end_time_ - last_base_pos_time));
        }
    }

}
