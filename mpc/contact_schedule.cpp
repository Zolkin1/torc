//
// Created by zolkin on 8/6/24.
//

#include "contact_schedule.h"

#include <bits/ranges_algo.h>

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
        frame_schedule_map[frame].emplace_back(start_time, stop_time);
    }

    bool ContactSchedule::InContact(const std::string& frame, double time) const {
        return std::ranges::any_of(frame_schedule_map.at(frame).begin(), frame_schedule_map.at(frame).end(),
            [time](const std::pair<double, double>& cs){return time <= cs.second && time >= cs.first;});
    }


} // namespace torc::mpc