//
// Created by zolkin on 2/12/25.
//

#ifndef STEP_PLANNER_H
#define STEP_PLANNER_H

#include "proxsuite/proxqp/dense/dense.hpp"

#include "contact_schedule.h"
#include "simple_trajectory.h"

namespace torc::step_planning {
    using vectorx_t = Eigen::VectorXd;
    using vector2_t = Eigen::Vector2d;
    using vector4_t = Eigen::Vector4d;
    using matrix2_t = Eigen::Matrix2d;
    using matrix3_t = Eigen::Matrix3d;
    using matrixx_t = Eigen::MatrixXd;
    using quat_t = Eigen::Quaterniond;

    class StepPlanner {
    public:
        StepPlanner(const std::vector<mpc::ContactInfo>& contact_polytopes, const std::vector<std::string>& contact_frames,
            const std::vector<double>& contact_offsets, double current_time_buffer, double polytope_buffer);

        /**
         * @brief Chooses the contact polytopes based on the raibert heuristic
         * @param q current state
         * @param v current velocity
         * @param q_target target configuration
         * @param v_target target velocity
         * @param contact_schedule [output] current contact schedule that will be updated with the new contact polytopes
         * @param nominal_footholds [output] nominal foothold based on raibert
         * @param projected_footholds [output] the foothold projected onto the polytope
         */
        void PlanStepsHeuristic(const mpc::SimpleTrajectory& q_target,
                                const std::vector<double>& dt_vec,
                                mpc::ContactSchedule& contact_schedule,
                                std::map<std::string, std::vector<vector2_t>>& nominal_footholds,
                                std::map<std::string, std::vector<vector2_t>>& projected_footholds,
                                bool first_loop = false);

        void UpdateContactPolytopes(const std::vector<mpc::ContactInfo>& contact_polytopes);

    protected:

        std::vector<double> ComputeContactMidtimes(const std::string& frame,
            const mpc::ContactSchedule& contact_schedule, double traj_end_time);

        static vectorx_t InterpolateTarget(const mpc::SimpleTrajectory& target, const std::vector<double>& dt_vec, double time);

        static bool InPolytope(const mpc::ContactInfo& polytope, const vector2_t& point);

        void SetFootTargetAndPolytope(double midtime, int contact_idx,
            const mpc::SimpleTrajectory& q_target, const std::vector<double>& dt_vec, int frame_idx,
            mpc::ContactSchedule& contact_schedule,
            std::map<std::string, std::vector<vector2_t>>& nominal_footholds,
            std::map<std::string, std::vector<vector2_t>>& projected_footholds);

        std::pair<vector2_t, int> ProjectOntoClosestPolytope(const vector2_t& point);

        std::pair<vector2_t, double> ProjectOntoPolytope(const vector2_t& point, const mpc::ContactInfo& polytope);

        std::vector<mpc::ContactInfo> contact_polytopes_;
        std::vector<std::string> contact_frames_;
        std::vector<vector2_t> contact_offsets_;
        double current_time_buffer_;    // Only update the polytopes that start after the current time buffer
        double polytope_buffer_;

        // ProxQpInterface
        proxsuite::proxqp::dense::QP<double> qp_;

    private:
    };
}



#endif //STEP_PLANNER_H
