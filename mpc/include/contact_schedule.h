//
// Created by zolkin on 8/6/24.
//

#ifndef CONTACT_SCHEDULE_H
#define CONTACT_SCHEDULE_H

#include <map>
#include <string>
#include <vector>

namespace torc::mpc {
    /**
     * @brief Holds a contact schedule.
     *
     * By default the frames are in contact, and therefore swing phases are inserted
     *
     * The times in the contact schedule are in relative time, i.e., the current time is always 0.
     * If you want to updated a contact schedule that was made earlier, you can use the ShiftContacts function.
     */
    class ContactSchedule {
        public:
        ContactSchedule() = default;
        explicit ContactSchedule(const std::vector<std::string>& frames);

        void SetFrames(const std::vector<std::string>& frames);
        void Reset();

        /**
         * @brief Inserts a contact into the schedule
         *
         * @param frame the frame in contact
         * @param start_time the start time of the contact
         * @param stop_time the end time of the contact
         */
        // void InsertContact(const std::string& frame, double start_time, double stop_time);

        void InsertSwing(const std::string& frame, double start_time, double stop_time);

        void InsertSwingByDuration(const std::string& frame, double start_time, double duration);

        [[nodiscard]] bool InContact(const std::string& frame, double time) const;

        [[nodiscard]] bool InSwing(const std::string& frame, double time) const;


        /**
         * @brief Shifts the entire contact schedule by a set amount.
         *
         * @param shift the amount of time to shift the contact schedule by. A positive shift will push the contacts
         *  out, while a negative shift will bring them in. If you want to simulate the contact schedule moving through
         *  time, you will need to apply a negative shift.
         */
        void ShiftSwings(double shift);

        /**
         * @brief Removes all contacts that end before time = time_cutoff
         */
        void CleanContacts(double time_cutoff);

        /**
         * @brief Creates a default swing trajectory given the contact schedule and information provided.
         *
         * In general this will only work
         *  on flat ground scenarios. This will generate trajectories for a given frame. This is mostly designed for feet
         *  although it may also work with hands. This function assigns the same swing traj to each swing in the trajectory.
         *
         *  The default swing trajectory is two cubic splines attached to each other. The start and end zdot are 0.
         *  The velocity is made constant throughout the trajectory.
         *  If the swing time has no end during the trajectory, then we assume it is the same length as the previous contact time.
         *
         *  The start height for the first swing is determine by the time in the swing.
         *
         * @param frame the frame to make the trajectory for
         * @param apex_height the apex height of the swing trajectory
         * @param end_height the ending height of the swing trajectory. This is also used for all in contact heights
         * @param apex_time the percentage through the swing during which the apex should be reached
         * @param dt_vec the list of dt's. The length of which should be the number of nodes in the MPC
         * @param swing_traj the swing trajectory to populate
         */
        void CreateSwingTraj(const std::string& frame, double apex_height, double end_height,
                             double apex_time, const std::vector<double>& dt_vec,  std::vector<double>& swing_traj) const;


        const std::map<std::string, std::vector<std::pair<double, double>>>&  GetScheduleMap() const;

     // double GetLastContactTime(const std::string& frame);
        double GetLastSwingTime(const std::string& frame) const;
    protected:
         static double GetTime(const std::vector<double>& dt_vec, int node);

         static double GetSwingHeight(double apex_height, double ground_height, double apex_time, double time,
             double start_time, double end_time);

        std::map<std::string, std::vector<std::pair<double, double>>> frame_schedule_map;
    };
}    // namespace torc::mpc


#endif //CONTACT_SCHEDULE_H
