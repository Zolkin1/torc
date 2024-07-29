//
// Created by zolkin on 6/18/24.
//

#ifndef TORC_TRAJECTORY_H
#define TORC_TRAJECTORY_H

//#include "robot_state_types.h"
//#include "robot_contact_info.h"
#include <Eigen/Core>

namespace torc::mpc {
    using vectorx_t = Eigen::VectorXd;
    using vector3_t = Eigen::Vector3d;

    class Trajectory {
    public:
        int GetNumNodes() const;
        void SetNumNodes(int nodes);

        void SetConfiguration(int node, const vectorx_t& q);
        void SetVelocity(int node, const vectorx_t& v);
        void SetTau(int node, const vectorx_t& tau);
        void SetForce(int node, const vectorx_t& f);

//        void Reset();

        // TODO: Add interpolation function

    protected:
    private:
        std::vector<vectorx_t> q_;
        std::vector<vectorx_t> v_;
        std::vector<vectorx_t> tau_;
        std::vector<vector3_t> forces_;
        int nodes_;
    };
}

#endif //TORC_TRAJECTORY_H
