//
// Created by zolkin on 1/18/25.
//

#ifndef HPIPM_MPC_H
#define HPIPM_MPC_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <filesystem>

#include "BoxConstraint.h"
#include "hpipm-cpp/hpipm-cpp.hpp"
#include "full_order_rigid_body.h"
#include "trajectory.h"
#include "constraint.h"
#include "DynamicsConstraint.h"
#include "InputConstraint.h"

#include "MpcSettings.h"
#include "StateConstraint.h"
#include "StateInputConstraint.h"

namespace torc::mpc {
    namespace fs = std::filesystem;

    using vectorx_t = Eigen::VectorXd;
    using vector2_t = Eigen::Vector2d;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3x_t = Eigen::Matrix3Xd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix43_t = Eigen::Matrix<double, 4, 3>;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    using sp_matrixx_t = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;

    class HpipmMpc {
    public:
        HpipmMpc(MpcSettings settings);

        void SetDynamicsConstraints(std::vector<DynamicsConstraint> constraints);

        void Compute();

        void UpdateSetttings(MpcSettings settings);
    protected:
        void CreateConstraints();

    private:
        std::vector<Constraint> constraints;
        MpcSettings settings_;

        std::vector<DynamicsConstraint> dynamics_constraints_;
        std::unique_ptr<BoxConstraint> config_box_;
        std::unique_ptr<BoxConstraint> vel_box_;
        std::unique_ptr<BoxConstraint> tau_box_;

        std::unique_ptr<InputConstraint> friction_cone_;

        std::unique_ptr<StateConstraint> swing_height_;

        std::unique_ptr<StateConstraint> holonomic_;
        std::unique_ptr<StateInputConstraint> holonomic_cent_;

        std::unique_ptr<StateConstraint> polytope_;

        std::unique_ptr<StateConstraint> collision_;

        // Solver
        std::vector<hpipm::OcpQp> qp;

        // Trajectories
        Trajectory traj_;
    };
}

#endif //HPIPM_MPC_H
