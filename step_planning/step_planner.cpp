//
// Created by zolkin on 2/12/25.
//
#include <iostream>
#include "torc_timer.h"
#include "step_planner.h"

#include <boost/mpl/aux_/numeric_op.hpp>

namespace torc::step_planning {
    StepPlanner::StepPlanner(const std::vector<mpc::ContactInfo> &contact_polytopes,
        const std::vector<std::string> &contact_frames, const std::vector<double> &contact_offsets,
        double current_time_buffer, double polytope_buffer, const std::string& log_file_name)
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

        // TODO: Why did I put this in? I think it was for something related to the initialization
        // contact_polytopes_.push_back(mpc::ContactSchedule::GetDefaultContactInfo());

        log_file_.open(log_file_name);
    }

    StepPlanner::~StepPlanner() {
        log_file_.close();
    }

    StepPlanner::StepPlanner(const std::vector<mpc::ContactInfo> &contact_polytopes,
        const std::vector<std::string> &contact_frames, const std::vector<double> &contact_offsets,
        double current_time_buffer, double polytope_buffer, const std::string& log_file_name, int seed) : StepPlanner(contact_polytopes,
            contact_frames, contact_offsets, current_time_buffer, polytope_buffer, log_file_name) {
        gen_.seed(seed);
    }


    void StepPlanner::UpdateContactPolytopes(const std::vector<mpc::ContactInfo> &contact_polytopes) {
        contact_polytopes_ = contact_polytopes;
    }

    void StepPlanner::PlanStepsHeuristic(const mpc::SimpleTrajectory &q_target,
        const std::vector<double>& dt_vec,
        mpc::ContactSchedule &contact_schedule,
        std::map<std::string, std::vector<vector2_t>>& nominal_footholds,
        std::map<std::string, std::vector<vector2_t>>& projected_footholds,
        double time,
        bool first_loop) {

        nominal_footholds.clear();
        projected_footholds.clear();

        double traj_end_time = 0;
        for (const auto& dt : dt_vec) {
            traj_end_time += dt;
        }

        for (int j = 0; j < contact_frames_.size(); j++) {
            const std::string frame = contact_frames_[j];

            log_file_ << 1 << "," << frame << ",R," << time << "," << contact_schedule.GetPolytopes(frame).size() << ","; // "R" for raibert

            nominal_footholds.insert({frame, {}});
            projected_footholds.insert({frame, {}});

            // Compute contact midtimes
            std::vector<double> midtimes = ComputeContactMidtimes(frame, contact_schedule, traj_end_time);
            if (midtimes.size() != contact_schedule.GetNumContacts(frame)) {
                throw std::runtime_error("[PlanStepsHeuristic] Computed midtimes size does not match contact schedule contact size!");
            }

            // Get target state at contact midtimes
            bool midtimes_negative = true;
            for (int i = 0; i < midtimes.size(); i++) {
                if (contact_schedule.InContact(frame, 0) && (first_loop || i > contact_schedule.GetContactIndex(frame, 0))) {
                    SetFootTargetAndPolytope(midtimes[i], i, q_target, dt_vec, j, contact_schedule,
                        nominal_footholds, projected_footholds);
                } else if (contact_schedule.InSwing(frame, 0) && midtimes[i] > current_time_buffer_) {
                    SetFootTargetAndPolytope(midtimes[i], i, q_target, dt_vec, j, contact_schedule,
                                            nominal_footholds, projected_footholds);
                }

                if (midtimes[i] > 0) {
                    midtimes_negative = false;
                }
                log_file_ << midtimes[i] + time << "," << GetPolytopeIdx(contact_schedule.GetPolytopes(frame)[i]) << ",";
            }
            log_file_ << std::endl;
        }
    }

    void StepPlanner::PlanStepsSampling(const mpc::SimpleTrajectory &q_target, const std::vector<double> &dt_vec,
        std::vector<mpc::ContactSchedule>& contact_schedule, std::map<std::string, std::vector<vector2_t> > &nominal_footholds,
        std::map<std::string, std::vector<vector2_t> > &projected_footholds,
        double time,
        bool first_loop) {
        nominal_footholds.clear();
        projected_footholds.clear();

        double traj_end_time = 0;
        for (const auto& dt : dt_vec) {
            traj_end_time += dt;
        }

        // Verify that the contact schedules are compatible (for now they must have the same number of contacts)
        for (const auto& frame : contact_frames_) {
            for (int j = 1; j < contact_schedule.size(); j++) {
                if (contact_schedule[j].GetNumContacts(frame) != contact_schedule[j-1].GetNumContacts(frame)) {
                    throw std::runtime_error("[StepPlanner] Contact schedule vector is not consistent! Contact number mis-match!");
                }
            }
        }

        std::map<double, std::vector<std::vector<int>>> sampled_polys_midtimes;

        // --------- Get the Midtimes --------- //
        // All the contact schedules are assumed to have the same timing for now
        mpc::ContactSchedule& sched = contact_schedule[0];
        std::vector<std::vector<double>> midtimes_all_frames;

        for (int j = 0; j < contact_frames_.size(); j++) {
            const std::string frame = contact_frames_[j];

            nominal_footholds.insert({frame, {}});
            projected_footholds.insert({frame, {}});

            // Compute contact midtimes
            midtimes_all_frames.push_back(ComputeContactMidtimes(frame, sched, traj_end_time));
            if (midtimes_all_frames.back().size() != sched.GetNumContacts(frame)) {
                throw std::runtime_error("[PlanStepsHeuristic] Computed midtimes size does not match contact schedule contact size!");
            }
        }
        // std::cerr << "All contact midtimes computed!" << std::endl;

        // Group the frames by having the same midtimes
        std::map<std::vector<double>, std::vector<std::string>> frame_groups;   // Indexed by midtime
        for (int j = 0; j < contact_frames_.size(); j++) {
            frame_groups[midtimes_all_frames[j]].push_back(contact_frames_[j]);
        }
        // std::cerr << "Frames grouped!" << std::endl;

        for (const auto& [midtimes, frames] : frame_groups) {
            for (const auto& mt : midtimes) {
                sampled_polys_midtimes.insert({mt, {}});    // Make elements for all midtimes
            }
        }
        // std::cerr << "sampled_polys created!" << std::endl;
        // ------------------------------------ //

        // Iterate through each contact schedule, one for each parallel MPC
        for (int sched_idx = 0; sched_idx < contact_schedule.size(); sched_idx++) {
            std::vector<int> sample_frame_idxs;
            std::vector<int> nominal_idx;

            for (const auto& [midtimes, frames] : frame_groups) {
                // Get target state at contact midtimes
                for (int i = 0; i < midtimes.size(); i++) {
                    if ((sched.InContact(frames[0], 0) && (first_loop || i > sched.GetContactIndex(frames[0], 0))) ||
                        sched.InSwing(frames[0], 0) && midtimes[i] > current_time_buffer_) {
                        // std::cerr << "Valid contact to sample! (midtime: " << midtimes[i] << ")(frame1: "
                            // << frames[0] << ", frame2: " << frames[1] << ")" << std::endl;
                        // std::cout << "i: " << i << std::endl;
                        // std::cout << "midtimes size: " << midtimes.size() << std::endl;
                        // std::cout << "num contacts: " << sched.GetNumContacts(frames[0]) << std::endl;
                        sampled_polys_midtimes[midtimes[i]].push_back(SetFootTargetAndPolytopeSampling(midtimes[i], i,
                            frames, q_target, dt_vec, sampled_polys_midtimes[midtimes[i]], sched,
                            nominal_footholds, projected_footholds));
                        // std::cerr << "Sample successful!" << std::endl;
                    }
                    // std::cout << "-----" << std::endl;
                }
            }
            // std::cerr << "Sampling completed!" << std::endl;

            for (int i = 0; i < contact_frames_.size(); i++) {
                const std::string frame = contact_frames_[i];
                log_file_ << sched_idx << "," << frame << ",S," << time << "," << sched.GetPolytopes(frame).size() << ","; // "S" for sampling
                for (int j = 0; j < midtimes_all_frames[i].size(); j++) {
                    log_file_ << midtimes_all_frames[i][j] + time << "," << GetPolytopeIdx(sched.GetPolytopes(frame)[j]) << ",";
                }
                log_file_ << std::endl;
            }

            // std::cerr << "Logging completed!" << std::endl;

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
        return (mat_res[0] <= polytope.b_[0] && mat_res[0] >= polytope.b_[2]) &&
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
        double min_time_diff = 1e10;
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

        // DEBUG CHECK
        if (midtime <= 0) {
            throw std::runtime_error("[StepPlanner] Midtime must be positive!");
        }

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
        bool not_in_any_polytope = true;
        for (const auto& poly : contact_polytopes_) {
            if (InPolytope(poly, nominal_footholds[frame].back())) {
                projected_footholds[frame].push_back(nominal_footholds[frame].back());  // Projected matches nominal

                // Update contact schedule
                contact_schedule.SetPolytope(frame, contact_idx, poly);
                // std::cout << "[StepPlanner] Polytope height" << poly.height_ << std::endl;

                not_in_any_polytope = false;
                // std::cerr << "In polytope: " << poly.b_.transpose() << std::endl;
                break;
            }
        }

        if (not_in_any_polytope) {
            int polytope_idx = -1;
            // Project onto closest polytope
            vector2_t projected_point;
            std::tie(projected_point, polytope_idx) = ProjectOntoClosestPolytope(nominal_footholds[frame].back());
            projected_footholds[frame].push_back(projected_point);

            // std::cerr << "Projected onto polytope: " << contact_polytopes_[polytope_idx].b_.transpose() << std::endl;

            // Update contact schedule
            contact_schedule.SetPolytope(frame, contact_idx, contact_polytopes_[polytope_idx]);
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

    std::vector<int> StepPlanner::SetFootTargetAndPolytopeSampling(double midtime, int contact_idx, const std::vector<std::string>& frames,
        const mpc::SimpleTrajectory &q_target, const std::vector<double> &dt_vec,
        const std::vector<std::vector<int> > &used_polys, mpc::ContactSchedule &contact_schedule,
        std::map<std::string, std::vector<vector2_t> > &nominal_footholds,
        std::map<std::string, std::vector<vector2_t> > &projected_footholds) {

        // DEBUG CHECK
        if (midtime <= 0) {
            throw std::runtime_error("[StepPlanner] Midtime must be positive!");
        }

        std::vector<int> sample_frame_idxs;
        std::map<std::string, int> nominal_idx;
        for (int i = 0; i < frames.size(); i++) {
            const std::string frame = frames[i];
            // std::cerr << "frame: " << frame << std::endl;

            nominal_idx.insert({frame, {}});

            int frame_idx = 0;
            for (int j = 0; j < contact_frames_.size(); j++) {
                if (frame == contact_frames_[j]) {
                    frame_idx = j;
                    // std::cerr << "frame at frame idx: " << contact_frames_[frame_idx] << std::endl;
                    break;
                }
            }

            if (contact_schedule.InContact(frame, midtime)) {
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
                bool not_in_any_polytope = true;
                for (const auto& poly : contact_polytopes_) {
                    if (InPolytope(poly, nominal_footholds[frame].back())) {
                        projected_footholds[frame].push_back(nominal_footholds[frame].back());  // Projected matches nominal

                        // Update contact schedule
                        contact_schedule.SetPolytope(frame, contact_idx, poly);
                        // std::cout << "[StepPlanner] Polytope height" << poly.height_ << std::endl;

                        not_in_any_polytope = false;
                        break;
                    }
                }

                if (not_in_any_polytope) {
                    sample_frame_idxs.push_back(frame_idx);
                    // std::cerr << "Just added " << contact_frames_[sample_frame_idxs.back()] << std::endl;
                    nominal_idx[frame] = nominal_footholds[frame].size()-1;
                }
            } else {
                throw std::runtime_error("[StepPlanner] Frame is not in contact!");
            }
        }

        std::vector<int> sampled_polys;

        // Sample the points that are not nominally in a polytope for this midtime
        if (!sample_frame_idxs.empty()) {
            // Get the points we need to sample around
            std::vector<vector2_t> points;
            for (int i = 0; i < sample_frame_idxs.size(); i++) {
                const std::string& frame = contact_frames_[sample_frame_idxs[i]];
                // std::cerr << frame << " at time " << midtime << " needs to be sampled!" << std::endl;
                // std::cerr << "point: " << nominal_footholds[frame][nominal_idx[frame]].transpose() << std::endl;
                points.push_back(nominal_footholds[frame][nominal_idx[frame]]);
                // DEBUG CHECK
                for (const auto& poly : contact_polytopes_) {
                    if (InPolytope(poly, points.back())) {
                        // std::cerr << "point: " << points.back().transpose() << std::endl;
                        // std::cerr << "nominal idx: " << nominal_idx[frame] << std::endl;
                        // std::cerr << "nominal foothold size: " << nominal_footholds.size() << std::endl;
                        throw std::runtime_error("[StepPlanner][DEBUG] Point is in polytope!");
                    }
                }
            }

            std::vector<std::pair<vector2_t, int>> projected_polytopes = SamplePolytopes(points, used_polys);
            for (const auto& [point, idx] : projected_polytopes) {
                sampled_polys.push_back(idx);
            }

            if (sampled_polys.empty()) {
                throw std::runtime_error("[StepPlanner] No sampled polytopes!");
            }

            assert(projected_polytopes.size() == sample_frame_idxs.size());

            if (sample_frame_idxs.size() != projected_polytopes.size()) {
                throw std::runtime_error("[StepPlanner] Number of sampled polytopes does not match the number of sample frames!");
            }

            for (int i = 0; i < projected_polytopes.size(); i++) {
                projected_footholds[contact_frames_[sample_frame_idxs[i]]].push_back(projected_polytopes[i].first);

                // Update contact schedule
                contact_schedule.SetPolytope(contact_frames_[sample_frame_idxs[i]], contact_idx, contact_polytopes_[projected_polytopes[i].second]);
            }
        }
        return sampled_polys;
    }


    int StepPlanner::GetPolytopeIdx(const mpc::ContactInfo &polytope) {
        for (int i = 0; i < contact_polytopes_.size(); i++) {
            const mpc::ContactInfo& planner_poly = contact_polytopes_[i];
            if (polytope.A_ == planner_poly.A_ && polytope.b_ == planner_poly.b_ && polytope.height_ == planner_poly.height_) {
                return i;
            }
        }

        if (polytope.A_ == mpc::ContactSchedule::GetDefaultContactInfo().A_ && polytope.b_ == mpc::ContactSchedule::GetDefaultContactInfo().b_) {
            throw std::runtime_error("[StepPlanner] getting the polytope idx for the default polytope!");
            return -1;
        }

        // std::cerr << "A: " << polytope.A_ << std::endl;
        // std::cerr << "b: " << polytope.b_ << std::endl;
        // std::cerr << "height: " << polytope.height_ << std::endl;
        throw std::runtime_error("[StepPlanner] provided polytope does not match any polytopes to select from!");
    }

    std::vector<std::pair<vector2_t, int>> StepPlanner::SamplePolytopes(const std::vector<vector2_t>& points,
        const std::vector<std::vector<int> > &used_polys) {
        double sample_rad = 0.25; // NOTE: Used to 0.25   // TODO: Read this in elsewhere

        // Create the sampling tree
        std::shared_ptr<SampleTreeNode> root = CreateSampleTree(points, sample_rad);

        // while (root->GetNumChildren() == 0) {
        //     sample_rad += 0.2;
        //     root = CreateSampleTree(points, sample_rad);
        // }

        // Prune previously used samples out of the tree
        for (int i = 0; i < used_polys.size(); i++) {
            // std::cerr << "used_polys[i]: " << "\t";
            // for (int i = 0; i < used_polys[i].size(); i++) {
            //     std::cerr << used_polys[i][i] << ", ";
            // }
            // std::cerr << std::endl;
            if (root->BranchExists(used_polys[i])) {    // Only check the unique used polys
                // Only prune brnaches that we took
                root->PruneBranch(used_polys[i]);
            }
        }

        std::vector<int> sampled_idxs;
        if (root->GetNumChildren() == 0) {
            // No available branches to sample
            // std::cerr << "[StepPlanner] No available unique sample paths to take!" << std::endl;    // TODO: Fix
            static int other_samples = 0;
            if (used_polys.empty()) {
                std::vector<std::pair<vector2_t, int>> sampled_polytopes;
                throw std::runtime_error("[StepPlanner] used_polys is empty!");
                return sampled_polytopes;
            }
            other_samples = other_samples%used_polys.size();
            for (int i = 0; i < used_polys[other_samples].size(); i++) {
                sampled_idxs.push_back(used_polys[other_samples][i]);
            }
            other_samples = (other_samples + 1);
        } else {
            // TODO: The way this is currently coded it will only work for two points at a time
            // Normalize the areas to 1
            double total_area = 0;
            for (int i = 0; i < root->GetNumChildren(); i++) {
                total_area += root->children[i]->area;
            }
            for (int i = 0; i < root->GetNumChildren(); i++) {
                root->children[i]->area /= total_area;
            }

            if (points.size() == 2) {
                for (int child_idx = 0; child_idx < root->GetNumChildren(); child_idx++) {
                    auto node = root->children[child_idx];
                    double total_area = 0;
                    for (int i = 0; i < node->GetNumChildren(); i++) {
                        total_area += node->children[i]->area;
                    }
                    for (int i = 0; i < node->GetNumChildren(); i++) {
                        node->children[i]->area /= total_area;
                    }
                }
            }

            // Sample
            // Generate a number between 0-1 from a given distribution
            std::uniform_real_distribution<double> dist(0., 1.);
            SampleTreeNode node = *root;
            for (int i = 0; i < points.size(); i++) {
                double sample = dist(gen_);  // Generate the samples
                // See which polytope it is in
                double start_area = 0;
                for (const auto& child : node.children) {
                    if (sample >= start_area && sample < child->area + start_area) {
                        // By construction, there will be a feasible sample
                        sampled_idxs.push_back(child->polytope_idx);
                        node = *child;
                        break;
                    }
                    start_area += child->area;
                }
            }
        }

        // Return
        std::vector<std::pair<vector2_t, int>> sampled_polytopes;
        for (int i = 0; i < sampled_idxs.size(); i++) {
            std::pair<vector2_t, int> s(ProjectOntoPolytope(points[i], contact_polytopes_[sampled_idxs[i]]).first,
                sampled_idxs[i]);
            sampled_polytopes.push_back(s);
        }
        return sampled_polytopes;
    }

    double StepPlanner::GetPolytopeCircleArea(const mpc::ContactInfo &polytope, const vector2_t &point, double radius) {

        // Polytope points
        std::vector<vector2_t> poly_points;
        poly_points.emplace_back(polytope.b_[0], polytope.b_[1]);
        poly_points.emplace_back(polytope.b_[0], polytope.b_[3]);
        poly_points.emplace_back(polytope.b_[2], polytope.b_[3]);
        poly_points.emplace_back(polytope.b_[2], polytope.b_[1]);
        // TODO: Might need to sort the polygon point counterclockwise

        // We don't support having the point inside the polytope (because then no sampling is needed)
        // vector2_t temp = polytope.A_ * point;
        // if ((temp(0) < polytope.b_[0] && temp(0) > polytope.b_[2]) &&
        //     (temp(1) < polytope.b_[1] && temp(1) > polytope.b_[3])) {
        //     throw std::runtime_error("[StepPlanner] Circle center in the polytope is unsupported!");
        // }

        if (InPolytope(polytope, point)) {
            std::cerr << "Point: " << point.transpose() << std::endl;
            throw std::runtime_error("[StepPlanner] Circle center in the polytope is unsupported!");
        }

        // There is an intersection, compute points
        bool has_intersection = false;
        std::vector<vector2_t> clipped_poly_points;
        for (int i = 0; i < poly_points.size(); i++) {
            std::vector<vector2_t> intersection;
            if (CircleLineIntersection(poly_points[i % 4], poly_points[(i+1) % 4], point, radius, intersection)) {
                for (const auto& inter : intersection) {
                    clipped_poly_points.push_back(inter);
                }
                has_intersection = true;
            }
        }

        if (!has_intersection) {
            bool inside_circle = true;
            for (int i = 0; i < poly_points.size(); i++) {
                if ((poly_points[i] - point).norm() > radius) {
                    inside_circle = false;
                }
            }

            if (inside_circle) {
                return ComputePolytopeArea(poly_points);
            }

            return 0;
        }

        if (clipped_poly_points.size() == 2) {
            double theta = acos((clipped_poly_points[0] - point).dot(clipped_poly_points[1] - point)/
                ((clipped_poly_points[0] - point).norm()*(clipped_poly_points[1] - point).norm()));
            // std::abs(std::atan2(std::abs(clipped_poly_points[0][1] - clipped_poly_points[1][1]),
            // std::abs(clipped_poly_points[0][0] - clipped_poly_points[1][0])));

            // Now grab all the polytope points inside the circle
            for (int i = 0; i < poly_points.size(); i++) {
                if ((poly_points[i] - point).norm() < radius) {
                    clipped_poly_points.push_back(poly_points[i]);
                }
            }

            // TODO: Might need to sort the polygon point counterclockwise
            double poly_area = ComputePolytopeArea(clipped_poly_points);
            double circ_segment_area = (radius*radius/2)*(theta - sin(theta));

            return poly_area + circ_segment_area;
        } else {
            // TODO: Fix up
            // For now just use the whole area
            return ComputePolytopeArea(poly_points);
        }
        // Determine the two points where the circle intersects the polytope

        // If there is an intersection then:
        //  (1) Create a new polytope with all the points inside the circle and the two intersection points. Compute the area of this polytope
        //      Computing the area of the polytope can be done with the shoelace formula
        //  (2) Compute the area of the circular segment
        //  (3) Sum those and return

        // If there is no intersection:
        // - Check if any point is in the circle. If that point is, then just compute the area of the polytope and return
        // - If no point is in the circle then return 0
    }

    std::shared_ptr<SampleTreeNode> StepPlanner::CreateSampleTree(const std::vector<vector2_t> &points, double sample_rad) {
        auto root = std::make_shared<SampleTreeNode>(-1, 0);
        // Create a circle around the point at is a rough approximation of the kinematic limits
        // Compute the area of each polytope in the circle
        for (int i = 0; i < contact_polytopes_.size(); i++) {
            double area = GetPolytopeCircleArea(contact_polytopes_[i], points[0], sample_rad);
            if (area > 0) {
                root->AddChild(i, area);
            }
        }

        // Normalize the areas to 1
        double total_area = 0;
        for (int i = 0; i < root->GetNumChildren(); i++) {
            total_area += root->children[i]->area;
        }

        for (int i = 0; i < root->GetNumChildren(); i++) {
            root->children[i]->area /= total_area;
        }

        if (points.size() > 1) {
            if (points.size() != 2) {
                throw std::runtime_error("[StepPlanner] currently can only make a sample tree for at most two points!");
            }

            for (int child_idx = 0; child_idx < root->GetNumChildren(); child_idx++) {
                std::shared_ptr<SampleTreeNode> node = root->children[child_idx];
                for (int point_idx = 1; point_idx < points.size(); point_idx++) {   // This isn't needed when points.size() == 2
                    // Create a circle around the point at is a rough approximation of the kinematic limits
                    // Compute the area of each polytope in the circle
                    for (int i = 0; i < contact_polytopes_.size(); i++) {
                        double area = GetPolytopeCircleArea(contact_polytopes_[i], points[point_idx], sample_rad);
                        if (area > 0) {
                            node->AddChild(i, area);
                        }
                    }
                }
            }
        }

        return root;
    }

    bool StepPlanner::CircleLineIntersection(const vector2_t &p1, const vector2_t &p2, const vector2_t &center,
        double rad, std::vector<vector2_t>& intersection) {
        // t^2a + t*b + c = 0
        double ax = center[0] - p1[0];
        double ay = center[1] - p1[1];
        double bx = p1[0] - p2[0];
        double by = p1[1] - p2[1];
        double a = std::pow(bx, 2) + std::pow(by, 2);
        double b = 2*(ax*bx + ay*by);
        double c = -rad*rad + ax*ax + ay*ay;

        if (b*b - 4*a*c > 0) {
            // Quadratic formula
            double t1 = (-b + std::sqrt(b*b - 4*a*c))/(2*a);
            double t2 = (-b - std::sqrt(b*b - 4*a*c))/(2*a);

            if ((t1 < 0 || t1 > 1) && (t2 < 0 || t2 > 1)) {
                return false;
            }

            if (t1 >= 0 && t1 <= 1) {
                intersection.push_back((1-t1)*p1 + t1*p2);
            }
            if (t2 >= 0 && t2 <= 1) {
                intersection.push_back((1-t2)*p1 + t2*p2);

            }
            return true;
        }
        return false;
    }

    double StepPlanner::ComputePolytopeArea(std::vector<vector2_t> points) {
        double area = 0;
        for (int i = 0; i < points.size(); i++) {
            vector2_t p1 = points[i];
            vector2_t p2 = points[(i + 1) % points.size()];

            area += (p1[0]*p2[1] - p2[0]*p1[1]);
        }

        return std::abs(area)*0.5;
    }


}