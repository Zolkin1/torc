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
#include "FrictionConeConstraint.h"
#include "HolonomicConstraint.h"

#include "MpcSettings.h"
#include "SwingConstraint.h"
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

    constexpr int FLOATING_BASE = 7;
    constexpr int FLOATING_VEL = 6;

    class HpipmMpc {
    public:
        HpipmMpc(MpcSettings settings, const models::FullOrderRigidBody& model);

        void SetDynamicsConstraints(std::vector<DynamicsConstraint> constraints);
        void SetConfigBox(const BoxConstraint& constraints);
        void SetVelBox(const BoxConstraint& constraints);
        void SetTauBox(const BoxConstraint& constraints);
        void SetFrictionCone(FrictionConeConstraint constraints);
        void SetSwingConstraint(SwingConstraint constraints);
        void SetHolonomicConstraint(HolonomicConstraint constraints);

        void CreateConstraints();
        void CreateCost();

        void Compute(const vectorx_t& q0, const vectorx_t& v0);

        void UpdateSetttings(MpcSettings settings);
    protected:

        /**
         * @brief
         * @param node
         * @return (row, col) pair
         */
        std::pair<int, int> GetFrictionIndex(int node);

        void ConvertQpSolToTraj();

    private:
        void SetSizes();

        void NanCheck();

        std::vector<Constraint> constraints;
        MpcSettings settings_;

        std::vector<DynamicsConstraint> dynamics_constraints_;
        std::unique_ptr<BoxConstraint> config_box_;
        std::unique_ptr<BoxConstraint> vel_box_;
        std::unique_ptr<BoxConstraint> tau_box_;

        std::unique_ptr<FrictionConeConstraint> friction_cone_;

        std::unique_ptr<SwingConstraint> swing_constraint_;

        std::unique_ptr<HolonomicConstraint> holonomic_;

        std::unique_ptr<SwingConstraint> polytope_;

        std::unique_ptr<SwingConstraint> collision_;

        // Solver
        std::vector<hpipm::OcpQp> qp;
        hpipm::OcpQpIpmSolverSettings qp_settings;
        std::vector<hpipm::OcpQpSolution> solution_;
        std::unique_ptr<hpipm::OcpQpIpmSolver> solver_;

        // Trajectories
        Trajectory traj_;

        // Sizes (per node)
        int ntau_;
        int nforces_;
        int nq_;
        int nv_;

        // Robot Model
        models::FullOrderRigidBody model_;
        int boundary_node_;

        // Swing
        std::vector<double> swing_traj_;
        std::vector<int> in_contact_;

        static constexpr int CONTACT_3DOF = 3;

        bool first_solve_;
    };
}

#endif //HPIPM_MPC_H
