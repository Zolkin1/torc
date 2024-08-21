//
// Created by zolkin on 8/6/24.
//

#include <cmath>
#include <algorithm>

#include "contact_schedule.h"

#include <iostream>


namespace torc::mpc {
    ContactSchedule::ContactSchedule(const std::vector<std::string>& frames) {
        SetFrames(frames);
    }


    void ContactSchedule::SetFrames(const std::vector<std::string>& frames) {
        frame_schedule_map.clear();
        for (const auto& frame : frames) {
            frame_schedule_map.insert({frame, {}});
        }
    }

    void ContactSchedule::Reset() {
        for (auto& cs : frame_schedule_map) {
            cs.second.clear();
        }
    }

    void ContactSchedule::InsertContact(const std::string& frame, double start_time, double stop_time) {
        // TODO: Consider verifying that there is no overlap with another contact
        frame_schedule_map[frame].emplace_back(start_time, stop_time);
    }

    bool ContactSchedule::InContact(const std::string& frame, double time) const {
        return std::ranges::any_of(frame_schedule_map.at(frame).begin(), frame_schedule_map.at(frame).end(),
            [time](const std::pair<double, double>& cs){return time <= cs.second && time >= cs.first;});
    }

    void ContactSchedule::ShiftContacts(double shift) {
        for (auto& [frame, contacts] : frame_schedule_map) {
            for (auto& [start, end] : contacts) {
                start += shift;
                end += shift;
            }
        }

        CleanContacts(-2);
    }

    void ContactSchedule::CleanContacts(double time_cutoff) {
        for (auto& [frame, contacts] : frame_schedule_map) {
            std::erase_if(contacts, [](const std::pair<double, double>& contact_pair) {return contact_pair.second < 0;});
        }
    }

    void ContactSchedule::CreateSwingTraj(const std::string& frame, double apex_height, double end_height,
        double apex_time, const std::vector<double>& dt_vec, std::vector<double>& swing_traj) const {
        if (apex_time < 0 || apex_time > 1) {
            throw std::invalid_argument("Apex time must be between 0 and 1!");
        }

        const int nodes = dt_vec.size();

        bool first_in_swing = true;
        double swing_start = 0;
        double swing_time = 0;
        double start_height = end_height;

        //Go through if its contact or not at each node
        for (int node = 0; node < nodes; node++) {
            // std::cout << "frame: " << frame << std::endl;
            // std::cout << "node: " << node << std::endl;
            // std::cout << "in contact: " << in_contact_[frame][node] << std::endl;
            double time = GetTime(dt_vec, node);
            if (InContact(frame, time)) {
                // In contact, set to the lowest height
                swing_traj[node] = end_height;
                first_in_swing = true; // Set to true for the next time it is in swing
            } else {
                if (first_in_swing) {
                    swing_start = GetTime(dt_vec, node);

                    // Determine when we next make contact
                    // Search for the next start after the current time
                    double smallest_start = 1e10;
                    for (const auto& [start, end] : frame_schedule_map.at(frame)) {
                        if (start > swing_start && start < smallest_start) {
                            smallest_start = start;
                        }
                    }

                    if (smallest_start != 1e10) {
                        swing_time = smallest_start - swing_start;
                    } else {
                        // Assume it continues for 0.2 seconds more since there is not another contact
                        swing_time = GetTime(dt_vec, nodes - 1) + 0.2 - swing_start;
                    }
                }

                if (swing_start == 0) {
                    // Get the largest negative end contact -- note that these are deleted with CleanContacts!
                    // Get the smallest start contact
                    double large_end = -1e10;
                    double small_start = 1e10;
                    for (const auto& [start, end] : frame_schedule_map.at(frame)) {
                        if (end < 0 && end > large_end) {
                            large_end = end;
                        }
                        if (start > 0 && start < small_start) {
                            small_start = start;
                        }
                    }

                    if (large_end == -1e10) {
                        large_end = 0;
                    }

                    if (small_start == 1e10) {
                        small_start = 0.2;
                    }

                    swing_time = small_start - large_end;

                    // Determine which part of the trajectory we are on
                    const double swing_start_real = large_end;

                    if (time < swing_time*apex_time + swing_start_real) {
                        // First half of the trajectory
                        swing_traj[node] = end_height
                            - std::pow(apex_time*swing_time, -2) * (3*(end_height - apex_height))*(std::pow(time - swing_start_real, 2))
                            + std::pow(apex_time*swing_time, -3) * 2*(end_height - apex_height) * std::pow(time - swing_start_real, 3);
                    } else {
                        // Second half
                        swing_traj[node] = apex_height
                            - std::pow(swing_time*(1 - apex_time), -2) * (3*(apex_height - end_height))*(std::pow(time - (apex_time*swing_time + swing_start_real), 2))
                            + std::pow(swing_time*(1 - apex_time), -3) * 2*(apex_height - end_height) * std::pow(time - (apex_time*swing_time + swing_start_real), 3);
                    }
                } else {
                    // Determine which spline to use
                    if (time < swing_time*apex_time + swing_start) {
                        // First half of trajectory
                        swing_traj[node] = end_height
                            - std::pow(apex_time*swing_time, -2) * (3*(end_height - apex_height))*(std::pow(time - swing_start, 2))
                            + std::pow(apex_time*swing_time, -3) * 2*(end_height - apex_height) * std::pow(time - swing_start, 3);
                    } else {
                        // Use the second half spline
                        swing_traj[node] = apex_height
                            - std::pow(swing_time*(1 - apex_time), -2) * (3*(apex_height - end_height))*(std::pow(time - (apex_time*swing_time + swing_start), 2))
                            + std::pow(swing_time*(1 - apex_time), -3) * 2*(apex_height - end_height) * std::pow(time - (apex_time*swing_time + swing_start), 3);
                    }
                }

                first_in_swing = false;
            }
        }
    }

    double ContactSchedule::GetTime(const std::vector<double>& dt_vec, int node) {
        double time = 0;
        if (node > dt_vec.size()) {
            throw std::runtime_error("[Contact schedule] Provided node is larger than the dt vector!");
        }

        for (int i = 0; i < node; i++) {
            time += dt_vec[i];
        }

        return time;
    }

    const std::map<std::string, std::vector<std::pair<double, double> > >& ContactSchedule::GetScheduleMap() const {
        return frame_schedule_map;
    }

    double ContactSchedule::GetLastContactTime(const std::string& frame) {
        double last_time = -1;
        for (const auto& [start, end] : frame_schedule_map[frame]) {
            if (end > last_time) {
                last_time = end;
            }
        }

        return last_time;
    }


} // namespace torc::mpc