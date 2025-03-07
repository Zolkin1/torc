//
// Created by zolkin on 12/2/24.
//

#ifndef REFERENCE_GENERATOR_H
#define REFERENCE_GENERATOR_H

#include <vector>
#include <Eigen/Core>

#include <proxsuite/proxqp/dense/dense.hpp>

#include "full_order_rigid_body.h"
#include "simple_trajectory.h"
#include "contact_schedule.h"

namespace torc::mpc {
    using vectorx_t = Eigen::VectorXd;
    using vector2_t = Eigen::Vector2d;
    using vector3_t = Eigen::Vector3d;
    using vector4_t = Eigen::Vector4d;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix2_t = Eigen::Matrix2d;

    /**
    * @brief Generate references for MPC given the user input and desired footholds.
    */
    class ReferenceGenerator {
    public:
        ReferenceGenerator(int nodes, const std::vector<std::string>& contact_frames,
            const std::vector<double>& dt, const models::FullOrderRigidBody& model, double polytope_delta);

        std::pair<SimpleTrajectory, SimpleTrajectory> GenerateReference(
            const vectorx_t& q,
            const vectorx_t& v,
            SimpleTrajectory q_target,
            SimpleTrajectory v_target,
            const std::map<std::string, std::vector<double>>& swing_traj,
            const std::vector<double>& hip_offsets,
            const ContactSchedule& contact_schedule,
            double target_height_offset, double current_ground_height,
            std::map<std::string, std::vector<vector3_t>>& contact_foot_pos,
            SimpleTrajectory& q_base_ref,
            SimpleTrajectory& v_base_ref);

    protected:
    private:
        /**
         * @brief Determines the node index for a given time.
         * @param time the time to associate with a node
         * @return the node index
         */
        int GetNode(double time);

        double GetTime(int node);

        vectorx_t GetCommandedConfig(double time, const SimpleTrajectory& q_target, const SimpleTrajectory& v_target);

        vectorx_t GetCommandedConfig(int node, const SimpleTrajectory& q_target);
        /**
         * @brief
         * @param time
         * @param times_and_bases must be sorted by time
         * @return
         */
        vectorx_t GetBasePositionInterp(double time, const std::vector<std::pair<double, vectorx_t>>& times_and_bases,
            const SimpleTrajectory& q_target,
            const vectorx_t& q_init);

        /**
         * @brief
         * @param foot_position the positition of the foot. This is modified if a projection occurs.
         * @param polytope
         * @return a boolean flag indicating if a projection took place
         */
        bool ProjectOnPolytope(vector2_t& foot_position, ContactInfo& polytope);

        vector2_t InterpolateBasePositions(int node, const std::map<double, vector2_t>& base_pos,
            const vector2_t &current_pos, const vector2_t& end_vel_command);

        int nodes_;
        int config_size_;
        int vel_size_;

        double polytope_delta_;

        std::vector<double> dt_;
        double end_time_;

        std::vector<std::string> contact_frames_;

        models::FullOrderRigidBody model_;

        // ProxQpInterface
        proxsuite::proxqp::dense::QP<double> qp_;

        static constexpr int FLOATING_BASE = 7;
        static constexpr int FLOATING_VEL = 6;
    };
}


#endif //REFERENCE_GENERATOR_H
