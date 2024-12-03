//
// Created by zolkin on 12/2/24.
//

#ifndef REFERENCE_GENERATOR_H
#define REFERENCE_GENERATOR_H

#include <vector>
#include <Eigen/Core>

#include "full_order_rigid_body.h"
#include "simple_trajectory.h"
#include "contact_schedule.h"

namespace torc::mpc {
    using vectorx_t = Eigen::VectorXd;
    using vector2_t = Eigen::Vector2d;
    using vector3_t = Eigen::Vector3d;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;

    /**
    * @brief Generate references for MPC given the user input and desired footholds.
    */
    class ReferenceGenerator {
    public:
        ReferenceGenerator(int nodes, int config_size, int vel_size, std::vector<std::string> contact_frames,
            std::vector<double> dt, std::shared_ptr<models::FullOrderRigidBody> model);

        std::pair<SimpleTrajectory, SimpleTrajectory> GenerateReference(const vectorx_t& q, const vectorx_t& v,
            const vector3_t& commanded_vel, const std::map<std::string, std::vector<double>>& swing_traj,
            const std::vector<double>& hip_offsets, const std::map<std::string, std::vector<int>>& in_contact,
            const ContactSchedule& contact_schedule);

    protected:
    private:
        /**
         * @brief Determines the node index for a given time.
         * @param time the time to associate with a node
         * @return the node index
         */
        int GetNode(double time);

        int nodes_;
        int config_size_;
        int vel_size_;

        std::vector<double> dt_;
        double end_time_;

        std::vector<std::string> contact_frames_;

        std::shared_ptr<models::FullOrderRigidBody> model_;
    };
}


#endif //REFERENCE_GENERATOR_H
