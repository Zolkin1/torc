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
#include "CentroidalDynamicsConstraint.h"
#include "CollisionConstraint.h"
#include "ConfigTrackingCost.h"
#include "hpipm-cpp/hpipm-cpp.hpp"
#include "full_order_rigid_body.h"
#include "trajectory.h"
#include "constraint.h"
#include "DynamicsConstraint.h"
#include "FrictionConeConstraint.h"
#include "HolonomicConstraint.h"
#include "LinearLsCost.h"
#include "contact_schedule.h"
#include "ForwardKinematicsCost.h"
#include "MpcSettings.h"
#include "PolytopeConstraint.h"
#include "SRBConstraint.h"
#include "SwingConstraint.h"

namespace torc::mpc {
    namespace fs = std::filesystem;

    using vectorx_t = Eigen::VectorXd;
    using vector2_t = Eigen::Vector2d;
    using vector3_t = Eigen::Vector3d;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3x_t = Eigen::Matrix3Xd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix43_t = Eigen::Matrix<double, 4, 3>;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    using sp_matrixx_t = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;

    constexpr int FLOATING_BASE = 7;
    constexpr int FLOATING_VEL = 6;

    enum HpipmLineSearchCondition {
        HConstraintViolation,
        HCostReduction,
        HBoth,
        HMinAlpha
    };

    class HpipmMpc {
    public:
        HpipmMpc(MpcSettings settings, const models::FullOrderRigidBody& model);
        ~HpipmMpc();

        void SetDynamicsConstraints(DynamicsConstraint constraint);
        void SetCentroidalDynamicsConstraints(CentroidalDynamicsConstraint constraint);
        void SetSrbConstraint(SRBConstraint constraint);
        void SetConfigBox(const BoxConstraint& constraints);
        void SetVelBox(const BoxConstraint& constraints);
        void SetTauBox(const BoxConstraint& constraints);
        void SetFrictionCone(FrictionConeConstraint constraints);
        void SetSwingConstraint(SwingConstraint constraints);
        void SetHolonomicConstraint(HolonomicConstraint constraints);
        void SetCollisionConstraint(CollisionConstraint constraints);
        void SetPolytopeConstraint(PolytopeConstraint constraint);

        // TODO: Write a version that takes in a SimpleTraj object
        void SetVelTrackingCost(LinearLsCost cost);
        void SetTauTrackingCost(LinearLsCost cost);
        void SetForceTrackingCost(LinearLsCost cost);
        void SetConfigTrackingCost(ConfigTrackingCost cost);
        void SetFowardKinematicsCost(ForwardKinematicsCost cost);

        void CreateConstraints();
        void CreateCost();

        // NOTE: The time is just for logging purposes
        hpipm::HpipmStatus Compute(double time, const vectorx_t& q0, const vectorx_t& v0, Trajectory& traj_out);

        void SetConfigTarget(const SimpleTrajectory& q_target);
        void SetVelTarget(const SimpleTrajectory& v_target);
        void SetForwardKinematicsTarget(const std::map<std::string, std::vector<vector3_t>>& fk_positions);

        void SetLinTraj(const Trajectory& traj_in);
        void SetLinTrajConfig(const SimpleTrajectory& config_traj);
        void SetLinTrajVel(const SimpleTrajectory& vel_traj);

        // Computes constraint violation for traj_
        double GetConstraintViolation(const std::vector<hpipm::OcpQpSolution>& sol, double alpha);
        double GetCost(const std::vector<hpipm::OcpQpSolution>& sol, double alpha);
        int GetSolveCounter() const;

        std::map<std::string, std::vector<double>> GetSwingTrajectory() const;

        void UpdateContactSchedule(const ContactSchedule& sched);
        void UpdateSetttings(MpcSettings settings);

        Trajectory GetTrajectory() const;

        void PrintNodeInfo() const;
    protected:

        /**
         * @brief
         * @param node
         * @return (row, col) pair
         */
        std::pair<int, int> GetFrictionIndex(int node);

        void ConvertQpSolToTraj(double alpha);

        // TODO: Make these const refs
        vectorx_t GetVelocityTarget(int node) const;
        vectorx_t GetTauTarget(int node) const;
        vector3_t GetForceTarget(int node, const std::string& frame) const;
        vectorx_t GetConfigTarget(int node) const;
        vector3_t GetEndEffectorTarget(int node, const std::string& frame) const;

        std::pair<double, double> LineSearch(const std::vector<hpipm::OcpQpSolution>& sol);

        double GetPolytopeConvergence(const std::string& frame, double time, const ContactSchedule& cs) const;

    private:
        void LogData(double time, const vectorx_t& q, const vectorx_t& v);

        void LogEigenVec(const vectorx_t& x);

        void SetSizes();

        void NanCheck();

        std::vector<Constraint> constraints;
        MpcSettings settings_;


        // --------- Constraints --------- //
        std::unique_ptr<DynamicsConstraint> dynamics_constraint_;
        std::unique_ptr<CentroidalDynamicsConstraint> centroidal_dynamics_constraint_;
        std::unique_ptr<SRBConstraint> srb_constraint_;
        std::unique_ptr<BoxConstraint> config_box_;
        std::unique_ptr<BoxConstraint> vel_box_;
        std::unique_ptr<BoxConstraint> tau_box_;

        std::unique_ptr<FrictionConeConstraint> friction_cone_;

        std::unique_ptr<SwingConstraint> swing_constraint_;

        std::unique_ptr<HolonomicConstraint> holonomic_;

        std::unique_ptr<PolytopeConstraint> polytope_;

        std::unique_ptr<CollisionConstraint> collision_;

        // --------- Cost --------- //
        std::unique_ptr<ConfigTrackingCost> config_tracking_;
        std::unique_ptr<LinearLsCost> vel_tracking_;
        std::unique_ptr<LinearLsCost> tau_tracking_;
        std::unique_ptr<LinearLsCost> force_tracking_;
        std::unique_ptr<ForwardKinematicsCost> fk_cost_;
        int fo_config_idx_;
        int fo_vel_idx_;
        int fo_force_idx_;
        int fo_tau_idx_;

        int cent_config_idx_;
        int cent_vel_idx_;
        int cent_force_idx_;

        int frame_tracking_idx_;

        // TODO: Consider allowing this to vary by node
        SimpleTrajectory v_target_;
        SimpleTrajectory q_target_;
        SimpleTrajectory tau_target_;

        std::map<std::string, std::vector<vector3_t>> end_effector_targets_;

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

        // Contacts & Swing
        std::map<std::string, std::vector<double>> swing_traj_;
        std::map<std::string, std::vector<int>> in_contact_;
        std::map<std::string, std::vector<ContactInfo>> contact_info_;

        static constexpr int CONTACT_3DOF = 3;

        bool first_constraint_gen_;
        bool first_cost_gen_;

        // Stats
        int solve_counter_;

        // Line search
        HpipmLineSearchCondition ls_condition_;
        double alpha_;

        // Logging
        std::ofstream log_file_;

    };
}

#endif //HPIPM_MPC_H
