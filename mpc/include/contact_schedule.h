//
// Created by zolkin on 8/6/24.
//

#ifndef CONTACT_SCHEDULE_H
#define CONTACT_SCHEDULE_H

#include <map>
#include <string>
#include <vector>

namespace torc::mpc {
    class ContactSchedule {
        public:
        ContactSchedule() = default;
        ContactSchedule(const std::vector<std::string>& frames);

        std::map<std::string, std::vector<std::pair<double, double>>> frame_schedule_map;

        void SetFrames(const std::vector<std::string>& frames);
        void Reset();
        void InsertContact(const std::string& frame, double start_time, double stop_time);
        [[nodiscard]] bool InContact(const std::string& frame, double time) const;
    };
}    // namespace torc::mpc


#endif //CONTACT_SCHEDULE_H
