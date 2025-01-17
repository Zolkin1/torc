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
        // A_default_ = matrixx_t::Identity(2, 2);
        // b_default_ = vector4_t::Constant(10);
    }


    void ContactSchedule::SetFrames(const std::vector<std::string>& frames) {
        frame_schedule_map.clear();

        for (const auto& frame : frames) {
            frame_schedule_map.insert({frame, {}});
            contact_polytopes.insert({frame, {}});
            contact_polytopes[frame].resize(frame_schedule_map[frame].size() + 1);
        }
    }

    void ContactSchedule::Reset() {
        for (auto& cs : frame_schedule_map) {
            cs.second.clear();
        }
    }

    // void ContactSchedule::InsertContact(const std::string& frame, double start_time, double stop_time) {
    //     // TODO: Consider verifying that there is no overlap with another contact
    //     frame_schedule_map[frame].emplace_back(start_time, stop_time);
    // }

    void ContactSchedule::InsertSwing(const std::string& frame, double start_time, double stop_time) {
        // TODO: Consider verifying that there is no overlap with another contact
        frame_schedule_map[frame].emplace_back(start_time, stop_time);
        contact_polytopes[frame].resize(frame_schedule_map[frame].size() + 1);
        contact_polytopes[frame].back() = GetDefaultContactInfo();
    }

    void ContactSchedule::InsertSwingByDuration(const std::string& frame, double start_time, double duration) {
        // TODO: Consider verifying that there is no overlap with another contact
        frame_schedule_map[frame].emplace_back(start_time, start_time + duration);
        contact_polytopes[frame].resize(frame_schedule_map[frame].size() + 1);
        contact_polytopes[frame].back() = GetDefaultContactInfo();
    }

    bool ContactSchedule::InContact(const std::string& frame, double time) const {
        return !std::ranges::any_of(frame_schedule_map.at(frame).begin(), frame_schedule_map.at(frame).end(),
            [time](const std::pair<double, double>& cs){return time <= cs.second && time >= cs.first;});
    }

    bool ContactSchedule::InSwing(const std::string& frame, double time) const {
       return !InContact(frame, time);
    }

    double ContactSchedule::GetSwingDuration(const std::string &frame, double time) const {
        if (InContact(frame, time)) {
            return -1;
        }

        int swing_idx = 0;
        while (frame_schedule_map.at(frame).at(swing_idx).second < time) {
            swing_idx++;
            // DEBUG CHECK
            if (swing_idx > frame_schedule_map.at(frame).size()) {
                throw std::runtime_error("[ContactSchedule::GetSwingDuration] swing idx too large!");
            }
        }

        return frame_schedule_map.at(frame).at(swing_idx).second - frame_schedule_map.at(frame).at(swing_idx).first;
    }

    double ContactSchedule::GetNextSwingDuration(const std::string &frame, double time) const {
        double min_start_diff = 100;
        std::pair<double, double> start_end;
        for (const auto& [start, end] : frame_schedule_map.at(frame)) {
            if (start > time && start - time < min_start_diff) {
                min_start_diff = start - time;
                start_end.first = start;
                start_end.second = end;
            }
        }

        return start_end.second - start_end.first;
    }

    double ContactSchedule::GetFirstContactTime(const std::string &frame) const {
        return frame_schedule_map.at(frame).at(0).second;
    }

    double ContactSchedule::GetSwingStartTime(const std::string &frame, double time) const {
        if (InContact(frame, time)) {
            return -1;
        }

        const auto& swings = frame_schedule_map.at(frame);
        int idx = swings.size() - 1;
        double start = swings.at(idx).first;
        if (swings[0].first > time) {
            throw std::runtime_error("[Contact schedule] error getting swing start time!");
        }

        while (start > time) {
            idx--;
            start = swings.at(idx).first;
        }

        return start;
    }


    void ContactSchedule::ShiftSwings(double shift) {
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
            std::erase_if(contacts, [time_cutoff](const std::pair<double, double>& contact_pair) {return contact_pair.second < time_cutoff;});
            contact_polytopes[frame].resize(GetNumContacts(frame));
        }
    }

    void ContactSchedule::CreateSwingTraj(const std::string& frame, double apex_height, double height_offset,
        double apex_time, const std::vector<double>& dt_vec, std::vector<double>& swing_traj) const {
        if (apex_time < 0 || apex_time > 1) {
            throw std::invalid_argument("Apex time must be between 0 and 1!");
        }

        const int nodes = dt_vec.size();
        swing_traj.resize(nodes);

        for (int node = 0; node < nodes; node++) {
            double time = GetTime(dt_vec, node);

            // Check if we are in swing
            if (InSwing(frame, time)) {
                for (const auto& [start, end] : frame_schedule_map.at(frame)) {
                    if (time >= start && time <= end) {
                        // TODO: Need to fix this! I am getting weird heights when stepping on the stairs
                        const int next_contact_idx = GetContactIndex(frame, end + 0.001);
                        const double end_height = contact_polytopes.at(frame).at(next_contact_idx).height_ + height_offset;

                        //TODO: I think I am seeing something weird here when stepping down stairs
                        const int prev_contact_idx = GetContactIndex(frame, start - 0.001);
                        const double start_height = contact_polytopes.at(frame).at(prev_contact_idx).height_ + height_offset;

                        const double apex_height_adjusted = apex_height + std::max(end_height, start_height);

                        swing_traj[node] = GetSwingHeight(apex_height_adjusted, end_height, start_height, apex_time, time, start, end);
                        break;
                    }
                }
            }
            if (!InSwing(frame, time)) {
                const int contact_idx = GetContactIndex(frame, time);
                const double end_height = contact_polytopes.at(frame)[contact_idx].height_ + height_offset;
                swing_traj[node] = end_height;
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

    // double ContactSchedule::GetLastContactTime(const std::string& frame) {
    //     double last_time = -1;
    //     for (const auto& [start, end] : frame_schedule_map[frame]) {
    //         if (end > last_time) {
    //             last_time = end;
    //         }
    //     }
    //
    //     return last_time;
    // }

    double ContactSchedule::GetLastSwingTime(const std::string& frame) const {
        double last_time = -1;
        for (const auto& [start, end] : frame_schedule_map.at(frame)) {
            if (end > last_time) {
                last_time = end;
            }
        }

        return last_time;
    }

    double ContactSchedule::GetSwingHeight(double apex_height, double end_height, double start_height, double apex_time, double time,
        double start_time, double end_time) {
        if (time < start_time) {
            std::cerr << "start time: " << start_time << std::endl;
            std::cerr << "end time: " << end_time << std::endl;
            std::cerr << "time: " << time << std::endl;
            throw std::runtime_error("[Contact Schedule] Provided time is before the start of the swing!");
        }
        if (time > end_time) {
            std::cerr << "start time: " << start_time << std::endl;
            std::cerr << "end time: " << end_time << std::endl;
            std::cerr << "time: " << time << std::endl;
            throw std::runtime_error("[Contact Schedule] Provided time is after the end of the swing!");
        }

        const double apex_time_abs = apex_time*(end_time - start_time) + start_time;

        const double swing_time = end_time - start_time;


        if (time < apex_time_abs) {
            // First spline
            return start_height
                    - std::pow(apex_time*swing_time, -2) * 3 * (start_height - apex_height) * (std::pow(time - start_time, 2))
                    + std::pow(apex_time*swing_time, -3) * 2 * (start_height - apex_height) * std::pow(time - start_time, 3);
        } else {
            // Second spline
            return apex_height
                    - std::pow(end_time - apex_time_abs, -2) * 3 * (-end_height + apex_height) * std::pow(time - apex_time_abs, 2)
                    + std::pow(end_time - apex_time_abs, -3) * 2 * (-end_height + apex_height) * std::pow(time - apex_time_abs, 3);
        }
    }

    std::vector<ContactInfo> ContactSchedule::GetPolytopes(const std::string& frame) const {
        return contact_polytopes.at(frame);
    }

    void ContactSchedule::SetPolytope(const std::string& frame, int contact_num, const matrixx_t& A, const vector4_t& b, double height) {
        if (contact_num >= GetNumContacts(frame)) {
            throw std::runtime_error("[Contact schedule] Invalid contact num!");
        }
        contact_polytopes[frame][contact_num].A_ = A;
        contact_polytopes[frame][contact_num].b_ = b;
        contact_polytopes[frame][contact_num].height_ = height;
    }

    int ContactSchedule::GetNumContacts(const std::string& frame) const {
    // TODO: This is not correct over the time horizon
        return frame_schedule_map.at(frame).size() + 1;
    }

    ContactInfo ContactSchedule::GetDefaultContactInfo() const {
        ContactInfo contact_info;

        // TODO: Come back to these defaults
        contact_info.A_ = matrixx_t::Identity(2, 2);
        contact_info.b_ << 100, 100, -100, -100;
        contact_info.height_ = 0;

        return contact_info;
    }

    int ContactSchedule::GetContactIndex(const std::string& frame, double time) const {
        // Loop through the swing times for the given frame
        // Get the contact index for the stuff between the swings
        if (InSwing(frame, time)) {
            throw std::runtime_error("[ContactSchedule::GetContactIndedx] Time provided is in a swing!");
        }
        const auto& swings = frame_schedule_map.at(frame);
        for (int i = 0; i < swings.size(); i++) {
            if (swings[i].first > time) {
                return i;
            }
        }

        return swings.size();
    }

    double ContactSchedule::GetInterpolatedHeight(const std::string &frame, double time) const {
        if (InContact(frame, time)) {
            const int contact_idx = GetContactIndex(frame, time);
            return contact_polytopes.at(frame)[contact_idx].height_;
        }

        const double start = GetSwingStartTime(frame, time);
        const double end = start + GetSwingDuration(frame, time);

        const int prev_contact_idx = GetContactIndex(frame, start - 0.001);
        const int next_contact_idx = GetContactIndex(frame, end + 0.001);
        const double h1 = contact_polytopes.at(frame)[prev_contact_idx].height_;
        const double h2 = contact_polytopes.at(frame)[next_contact_idx].height_;

        return h1 + (h2 - h1) * (time - start) / (end - start);
    }


} // namespace torc::mpc