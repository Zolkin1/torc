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

    StepPlanner::StepPlanner(const std::vector<mpc::ContactInfo> &contact_polytopes,
        const std::vector<std::string> &contact_frames, const std::vector<double> &contact_offsets,
        double current_time_buffer, double polytope_buffer, int seed) : StepPlanner(contact_polytopes,
            contact_frames, contact_offsets, current_time_buffer, polytope_buffer) {
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
        bool first_loop) {

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
            }
        }
    }

    void StepPlanner::PlanStepsSampling(const mpc::SimpleTrajectory &q_target, const std::vector<double> &dt_vec,
        std::vector<mpc::ContactSchedule>& contact_schedule, std::map<std::string, std::vector<vector2_t> > &nominal_footholds,
        std::map<std::string, std::vector<vector2_t> > &projected_footholds, bool first_loop) {
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

        // Iterate through each contact schedule, one for each parallel MPC
        for (int k = 1; k < contact_schedule.size(); k++) {
            mpc::ContactSchedule& sched = contact_schedule[k];

            for (int j = 0; j < contact_frames_.size(); j++) {
                const std::string frame = contact_frames_[j];

                nominal_footholds.insert({frame, {}});
                projected_footholds.insert({frame, {}});

                // Compute contact midtimes
                std::vector<double> midtimes = ComputeContactMidtimes(frame, sched, traj_end_time);
                if (midtimes.size() != sched.GetNumContacts(frame)) {
                    throw std::runtime_error("[PlanStepsHeuristic] Computed midtimes size does not match contact schedule contact size!");
                }

                // Get target state at contact midtimes
                for (int i = 0; i < midtimes.size(); i++) {
                    // Get all the previously sampled polytope combos
                    std::vector<std::vector<int>> sampled_polys;
                    for (int kk = 0; kk < k; kk++) {    // Go through all the contact schedules we have already sampled
                        sampled_polys.push_back({});    // Make a vector for each contact schedule we have already sampled

                        // Go through all the end effector frames looking for the ones in swing at the given mid-time
                        for (const auto& frame_inner : contact_frames_) {
                            if (contact_schedule[kk].InContact(frame_inner, midtimes[i])) {
                                // We are in contact, so add the corresponding polytope to the sampled polytopes
                                sampled_polys.back().push_back(GetPolytopeIdx(contact_schedule[kk].GetPolytopes(frame)[i]));
                            }
                        }
                    }

                    if ((sched.InContact(frame, 0) && (first_loop || i > sched.GetContactIndex(frame, 0))) ||
                        sched.InSwing(frame, 0) && midtimes[i] > current_time_buffer_) {
                        // TODO: This does NOT need to be called for every frame, just for every midtime.
                        SetFootTargetAndPolytopeSampling(midtimes[i], i, q_target, dt_vec, sampled_polys, sched,
                            nominal_footholds, projected_footholds);
                    }
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
                break;
            }
        }

        if (not_in_any_polytope) {
            int polytope_idx = -1;
            // Project onto closest polytope
            vector2_t projected_point;
            std::tie(projected_point, polytope_idx) = ProjectOntoClosestPolytope(nominal_footholds[frame].back());
            projected_footholds[frame].push_back(projected_point);

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

    void StepPlanner::SetFootTargetAndPolytopeSampling(double midtime, int contact_idx,
        const mpc::SimpleTrajectory &q_target, const std::vector<double> &dt_vec,
        const std::vector<std::vector<int> > &used_polys, mpc::ContactSchedule &contact_schedule,
        std::map<std::string, std::vector<vector2_t> > &nominal_footholds,
        std::map<std::string, std::vector<vector2_t> > &projected_footholds) {
        throw std::runtime_error("[StepPlanner] The sampling stuff needs to be thourghly tested!");

        // DEBUG CHECK
        if (midtime <= 0) {
            throw std::runtime_error("[StepPlanner] Midtime must be positive!");
        }

        std::vector<int> sample_frame_idxs;
        std::vector<int> nominal_idx;
        for (int i = 0; i < contact_frames_.size(); i++) {
            const std::string frame = contact_frames_[i];

            if (contact_schedule.InContact(frame, midtime)) {
                // Get base target
                vectorx_t target_state = InterpolateTarget(q_target, dt_vec, midtime);

                // Rotate into the correct frame & add offset
                vector4_t quat_vec = target_state.segment<4>(3);
                quat_vec.normalize();
                const quat_t quat(quat_vec);
                const matrix3_t R = quat.toRotationMatrix();

                target_state.head<2>() += R.topLeftCorner<2,2>()*contact_offsets_[i];
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
                    sample_frame_idxs.push_back(i);
                    nominal_idx.push_back(nominal_footholds[frame].size()-1);
                }
            }
        }

        if (!sample_frame_idxs.empty()) {
            // Get the points we need to sample around
            std::vector<vector2_t> points;
            for (int i = 0; i < sample_frame_idxs.size(); i++) {
                points.push_back(nominal_footholds[contact_frames_[sample_frame_idxs[i]]][nominal_idx[i]]);
            }

            int polytope_idx = -1;
            vector2_t projected_point;
            std::vector<std::pair<vector2_t, int>> projected_polytopes = SamplePolytopes(points, used_polys);

            for (int i = 0; i < sample_frame_idxs.size(); i++) {
                projected_footholds[contact_frames_[sample_frame_idxs[i]]].push_back(projected_point);

                // Update contact schedule
                contact_schedule.SetPolytope(contact_frames_[sample_frame_idxs[i]], contact_idx, contact_polytopes_[polytope_idx]);
            }
        }

    }


    int StepPlanner::GetPolytopeIdx(const mpc::ContactInfo &polytope) {
        for (int i = 0; i < contact_polytopes_.size(); i++) {
            mpc::ContactInfo& planner_poly = contact_polytopes_[i];
            if (polytope.A_ == planner_poly.A_ && polytope.b_ == planner_poly.b_ && polytope.height_ == planner_poly.height_) {
                return i;
            }
        }

        throw std::runtime_error("[StepPlanner] provided polytope does not match any polytopes to select from!");
    }

    std::vector<std::pair<vector2_t, int>> StepPlanner::SamplePolytopes(const std::vector<vector2_t>& points,
        const std::vector<std::vector<int> > &used_polys) {
        double sample_rad = 0.25;   // TODO: Read this in elsewhere

        // Create the sampling tree
        std::shared_ptr<SampleTreeNode> root = CreateSampleTree(points, sample_rad);

        // Prune previously used samples out of the tree
        for (int i = 0; i < used_polys.size(); i++) {
            root->PruneBranch(used_polys[i]);
        }

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
                for (int i = 0; i < root->GetNumChildren(); i++) {
                    node->children[i]->area /= total_area;
                }
            }
        }

        // Sample
        // Generate a number between 0-1 from a given distribution
        std::uniform_real_distribution<double> dist(0., 1.);
        std::vector<int> sampled_idxs;
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
        vector2_t temp = polytope.A_ * point;
        if ((temp(0) < polytope.b_[0] && temp(0) > polytope.b_[2]) &&
            (temp(1) < polytope.b_[1] && temp(1) > polytope.b_[3])) {
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

        if (clipped_poly_points.size() != 2) {
            throw std::runtime_error("[StepPlanner] got more than 2 circle-polytope intersections!");
        }

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