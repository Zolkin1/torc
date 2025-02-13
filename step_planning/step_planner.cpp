//
// Created by zolkin on 2/12/25.
//
#include <iostream>
#include "torc_timer.h"
#include "step_planner.h"

namespace torc::step_planning {
    StepPlanner::StepPlanner(const std::vector<mpc::ContactInfo> &contact_polytopes,
        const std::vector<std::string> &contact_frames, const std::vector<double> &contact_offsets,
        double current_time_buffer, double polytope_buffer)
            : contact_polytopes_(contact_polytopes), contact_frames_(contact_frames),
                current_time_buffer_(current_time_buffer), qp_(2,0,2), polytope_buffer_(polytope_buffer) {

        if (contact_offsets.size() != 2*contact_frames_.size()) {
            std::cerr << "Got contact offset size: " << contact_offsets_.size() << std::endl;
            std::cerr << "Expected contact offset size: " << 2*contact_frames_.size() << std::endl;
            throw std::runtime_error("Contact offset size does not match the contact frames size! Expecting double (x-y)!");
        }

        for (int i = 0; i < contact_offsets.size(); i+=2) {
            contact_offsets_.emplace_back(contact_offsets[i], contact_offsets[i+1]);
        }

    }

    void StepPlanner::UpdateContactPolytopes(const std::vector<mpc::ContactInfo> &contact_polytopes) {
        contact_polytopes_ = contact_polytopes;
    }

    void StepPlanner::PlanStepsHeuristic(const mpc::SimpleTrajectory &q_target,
        const std::vector<double>& dt_vec,
        mpc::ContactSchedule &contact_schedule,
        std::map<std::string, std::vector<vector2_t>>& nominal_footholds,
        std::map<std::string, std::vector<vector2_t>>& projected_footholds) {

        nominal_footholds.clear();
        projected_footholds.clear();

        double traj_end_time = 0;
        for (const auto& dt : dt_vec) {
            traj_end_time += dt;
        }

        for (int j = 0; j < contact_frames_.size(); j++) {
            const std::string frame = contact_frames_[j];
            nominal_footholds.insert({frame, {}});
            projected_footholds.insert({frame, {}});

            // Compute contact midtimes
            std::vector<double> midtimes = ComputeContactMidtimes(frame, contact_schedule, traj_end_time);
            if (midtimes.size() != contact_schedule.GetNumContacts(frame)) {
                throw std::runtime_error("[PlanStepsHeuristic] Computed midtimes size does not match contact schedule contact size!");
            }

            // Get target state at contact midtimes
            for (int i = 0; i < midtimes.size(); i++) {
                if (contact_schedule.InContact(frame, 0) && i > contact_schedule.GetContactIndex(frame, 0)) {
                    SetFootTargetAndPolytope(midtimes[i], i, q_target, dt_vec, j, contact_schedule,
                        nominal_footholds, projected_footholds);
                } else if (midtimes[i] > contact_schedule.GetSwingDuration(frame, 0)
                        + contact_schedule.GetSwingStartTime(frame, 0) && midtimes[i] > current_time_buffer_) {
                    SetFootTargetAndPolytope(midtimes[i], i, q_target, dt_vec, j, contact_schedule,
                                            nominal_footholds, projected_footholds);
                }
            }
        }
    }

    std::vector<double> StepPlanner::ComputeContactMidtimes(const std::string &frame,
        const mpc::ContactSchedule& contact_schedule, double traj_end_time) {
        std::vector<double> contact_midtimes;

        const auto& contact_map = contact_schedule.GetScheduleMap();
        const auto& swings = contact_map.at(frame);

        if (!swings.empty()) {
            double swing_time = swings[0].second - swings[0].first;

            // Handle the contact midpoint before the first swing
            contact_midtimes.emplace_back(swings[0].first - swing_time/2.0);
            // contact_midtimes[frame].emplace_back(std::max(0.0, swings[0].first - 0.15));
            for (int i = 0; i < swings.size() - 1; i++) {
                contact_midtimes.emplace_back((swings[i+1].first + swings[i].second)/2.0);
            }
            // Handle the contact midpoint after the last swing
            contact_midtimes.emplace_back(swings[swings.size() - 1].second + swing_time/2.0);
        } else {
            contact_midtimes.emplace_back(traj_end_time/2.0);
        }

        // DEBUG CHECK
        if (contact_schedule.GetPolytopes(frame).size() != contact_midtimes.size()) {
            throw std::runtime_error("[Reference generator] Polytopes size != contact_midtimes.size()");
        }
        // DEBUG CHECK
        if (contact_midtimes.size() != contact_schedule.GetNumContacts(frame)) {
            std::cerr << "frame: " << frame << std::endl;
            std::cerr << "contact_midtimes size: " << contact_midtimes.size() << std::endl;
            std::cerr << "num contacts: " << contact_schedule.GetNumContacts(frame) << std::endl;
            throw std::runtime_error("[Reference generator] NumContacts != contact_midtimes.size()");
        }

        std::map<std::string, int> polytope_idx_offset;     // Account for the deleted times when accessing the polytopes

        // // Remove all negative time contacts
        // polytope_idx_offset.insert({frame, 0});
        // for (int i = 0; i < contact_midtimes.size(); i++) {
        //     if (contact_midtimes[i] < 0) {
        //         contact_midtimes.erase(contact_midtimes.begin() + i);
        //         i--;
        //         polytope_idx_offset[frame]++;
        //     }
        // }

        return contact_midtimes;
    }

    bool StepPlanner::InPolytope(const mpc::ContactInfo &polytope, const vector2_t &point) {
        vector2_t mat_res = polytope.A_*point;
        return (mat_res[0] <= polytope.b_[0] && mat_res[1] >= polytope.b_[2]) &&
            (mat_res[1] <= polytope.b_[1] && mat_res[1] >= polytope.b_[3]);
    }


    vectorx_t StepPlanner::InterpolateTarget(const mpc::SimpleTrajectory &target,
        const std::vector<double>& dt_vec, double time) {
        // // TODO: Double check this!
        // double traj_end_time = 0;
        // for (int i = 0; i < dt_vec.size() - 1; i++) {
        //     traj_end_time += dt_vec[i];
        // }
        //
        // if (time > traj_end_time) {
        //     return target[target.GetNumNodes() - 1].head<2>();
        // }
        //
        // // Find the two vectors to interpolate between
        // double cumulative_time = 0.0;
        // int index1 = -1, index2 = -1;
        // for (size_t i = 0; i < dt_vec.size(); ++i) {
        //     cumulative_time += dt_vec[i];
        //     if (cumulative_time >= time) {
        //         index1 = i;
        //         if (i + 1 < dt_vec.size()) {
        //             index2 = i + 1; // Set index2 to the next vector
        //         } else {
        //             throw std::runtime_error("Cannot interpolate: the time t exceeds the range of the provided time deltas.");
        //         }
        //         break;
        //     }
        // }
        //
        // if (index1 == -1 || index2 == -1) {
        //     std::cerr << "Invalid time t for interpolation." << std::endl;
        //     exit(1);
        // }
        //
        // // Compute the interpolation factor
        // double alpha = (time - (cumulative_time - dt_vec[index1])) / dt_vec[index1];
        //
        // // Interpolate between vecs[index1] and vecs[index2]
        // return target[index1] + alpha * (target[index2] - target[index1]);

        // For now, to approximate, just take the one that is closest, no interpolation
        double min_time_diff = 100;
        int time_idx = -1;
        double traj_time = 0;
        for (int i = 0; i < dt_vec.size(); i++) {
            if (std::abs(time - traj_time) < min_time_diff) {
                min_time_diff = std::abs(time - traj_time);
                time_idx = i;
            }
            traj_time += dt_vec[i];
        }

        if (time_idx == -1) {
            throw std::runtime_error("[StepPlanner] Could not interpolate target!");
        }

        return target[time_idx];
    }

    void StepPlanner::SetFootTargetAndPolytope(double midtime, int contact_idx,
        const mpc::SimpleTrajectory &q_target, const std::vector<double>& dt_vec, int frame_idx,
        mpc::ContactSchedule& contact_schedule,
        std::map<std::string, std::vector<vector2_t>>& nominal_footholds,
        std::map<std::string, std::vector<vector2_t>>& projected_footholds) {

        const std::string frame = contact_frames_[frame_idx];

        // Get base target
        vectorx_t target_state = InterpolateTarget(q_target, dt_vec, midtime);

        // Rotate into the correct frame & add offset
        vector4_t quat_vec = target_state.segment<4>(3);
        quat_vec.normalize();
        const quat_t quat(quat_vec);
        const matrix3_t R = quat.toRotationMatrix();

        target_state.head<2>() += R.topLeftCorner<2,2>()*contact_offsets_[frame_idx];
        nominal_footholds[frame].push_back(target_state.head<2>());

        // Check if we are in a polytope
        bool not_in_polytope = false;
        for (const auto& poly : contact_polytopes_) {
            if (!InPolytope(poly, nominal_footholds[frame].back())) {
                not_in_polytope = true;
            } else {
                projected_footholds[frame].push_back(nominal_footholds[frame].back());  // Projected matches nominal

                // Update contact schedule
                contact_schedule.SetPolytope(frame, contact_idx, poly.A_, poly.b_);
            }
        }

        if (not_in_polytope) {
            // Project onto closest polytope
            const auto [projected_point, contact_idx] = ProjectOntoClosestPolytope(nominal_footholds[frame].back());
            projected_footholds[frame].push_back(projected_point);

            // Update contact schedule
            contact_schedule.SetPolytope(frame, contact_idx, contact_polytopes_[contact_idx].A_, contact_polytopes_[contact_idx].b_);
        }
    }

    std::pair<vector2_t, int> StepPlanner::ProjectOntoClosestPolytope(const vector2_t &point) {
        std::vector<double> distances;
        std::vector<vector2_t> projected_points;
        for (int i = 0; i < contact_polytopes_.size(); i++) {
            // TODO: Can I use a heuristic to not compute distances to ALL the polytopes?
            const auto [proj_point, dist] = ProjectOntoPolytope(point, contact_polytopes_[i]);
            distances.push_back(dist);
            projected_points.push_back(proj_point);
        }

        double min_distance = 10000.;
        int dist_idx = -1;
        for (int i = 0; i < contact_polytopes_.size(); i++) {
            if (distances[i] < min_distance) {
                min_distance = distances[i];
                dist_idx = i;
            }
        }

        if (dist_idx == -1) {
            throw std::runtime_error("[StepPlannet] error finding the closest polytope!");
        }

        return {projected_points[dist_idx], dist_idx};
    }

    std::pair<vector2_t, double> StepPlanner::ProjectOntoPolytope(const vector2_t &point, const mpc::ContactInfo &polytope) {
        matrix2_t H = 2*matrix2_t::Identity();
        vector2_t g = -2*point;

        matrixx_t Aeq(0,0);
        vectorx_t beq(0);

        vector4_t polytope_margin;
        polytope_margin << 1, 1, -1, -1;
        polytope_margin *= polytope_buffer_;
        vector4_t b_modified = polytope.b_ - polytope_margin;

        vector2_t lb, ub;
        lb << std::min(b_modified(0), b_modified(2)), std::min(b_modified(1), b_modified(3));
        ub << std::max(b_modified(0), b_modified(2)), std::max(b_modified(1), b_modified(3));

        torc::utils::TORCTimer qp_timer;
        qp_timer.Tic();

        qp_.init(H, g, Aeq, beq, polytope.A_, lb, ub);

        qp_.solve();

        qp_timer.Toc();

        if (std::abs(qp_.results.info.objValue - ((point - qp_.results.x).squaredNorm() - point.squaredNorm())) > 1e-4) {
            std::cerr << "got: " << qp_.results.info.objValue << std::endl;
            std::cerr << "expected: " << (point - qp_.results.x).squaredNorm() - point.squaredNorm() << std::endl;
            throw std::runtime_error("[Reference Generator] qp not formed correctly!");
        }

        return {qp_.results.x, qp_.results.info.objValue};
    }



}