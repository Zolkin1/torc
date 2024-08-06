//
// Created by zolkin on 7/28/24.
//
#ifndef TORC_FULL_ORDER_MPC_H
#define TORC_FULL_ORDER_MPC_H

#include <filesystem>

#include <Eigen/Core>

#include "osqp++.h"
#include "constraint.h"
#include "full_order_rigid_body.h"
#include "trajectory.h"
#include "explicit_fn.h"
#include "autodiff_fn.h"

#include "cost_function.h"


namespace torc::mpc {
    namespace fs = std::filesystem;

    using vectorx_t = Eigen::VectorXd;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3x_t = Eigen::Matrix3Xd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix43_t = Eigen::Matrix<double, 4, 3>;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    using sp_matrixx_t = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;

    // TODO:
    //  - Fix the orientation integration step (make sure the orientation is updated correctly)
    //  - Line search (probably do more cost function verification)
    //  - Setting swing trajectory
    //  - Setting q target and v target
    //  - Setting contact schedule

    struct MpcStats {
        osqp::OsqpExitCode solve_status;    // Exit code from solver
        double qp_cost;                     // OSQP Cost
        double full_cost;                   // Full nonlinear cost
        double alpha;                       // Linesearch alpha value
        double qp_res_norm;                 // Norm of the QP result vector
        double total_compute_time;          // Time for the entire Compute function
    };

    class FullOrderMpc {
    public:
        FullOrderMpc(const fs::path& config_file, const fs::path& model_path);

        /**
         * @brief Allocate the large sections of memory, setup the QP solver.
         *
         * Create the sparse eigen matrix with the correct sparsity pattern.
         * Pass to OSQP to allocate memory and perform KKT factorization.
         */
        void Configure();

        /**
         * @brief Update the contact schedule. This sets the internal in or out of contact flags.
         *
         * Also can perform any time discretization scaling.
         */
        void UpdateContactSchedule();

        /**
         * @brief Computes the trajectory given the current state.
         *
         * TODO: needs to reset the triplet_idx_ at some point
         *
         * @param state
         * @return
         */
        void Compute(const vectorx_t& state, Trajectory& traj_out);

        void SetVerbosity(bool verbose);
        [[nodiscard]] std::vector<std::string> GetContactFrames() const;
        [[nodiscard]] int GetNumNodes() const;
        void PrintStatistics() const;

        void SetWarmStartTrajectory(const Trajectory& traj);
    protected:
        enum ConstraintType {
        Integrator,
        ID,
        FrictionCone,
        ConfigBox,
        VelBox,
        TorqueBox,
        SwingHeight,
        Holonomic
        };

        enum DecisionType {
            Configuration,
            Velocity,
            Torque,
            GroundForce
        };

        struct Workspace {
            matrixx_t int_mat;
            matrixx_t id_config_mat;
            matrixx_t id_vel1_mat;
            matrixx_t id_vel2_mat;
            matrixx_t id_force_mat;
            matrixx_t fric_cone_mat;
            vectorx_t swing_vec;
            matrix6x_t frame_jacobian;
            matrixx_t holo_mat;
            vectorx_t acc;
            matrixx_t obj_config_mat;
            matrixx_t obj_vel_mat;
            // TODO: Consider combining these two vectors
            vectorx_t obj_config_vector;
            vectorx_t obj_vel_vector;
            std::vector<models::ExternalForce> f_ext;
        };

    // -------- Constraint Creation -------- //
        void CreateConstraints();
        void AddICConstraint();
        void AddIntegrationConstraint(int node);
        void AddIDConstraint(int node);
        void AddFrictionConeConstraint(int node);
        void AddConfigurationBoxConstraint(int node);
        void AddVelocityBoxConstraint(int node);
        void AddTorqueBoxConstraint(int node);
        void AddSwingHeightConstraint(int node);
        void AddHolonomicConstraint(int node);

    // -------- Linearization Helpers ------- //
        matrix3_t QuatIntegrationLinearizationXi(int node);
        matrix3_t QuatIntegrationLinearizationW(int node);

        void InverseDynamicsLinearization(int node, matrixx_t& dtau_dq, matrixx_t& dtau_dv1, matrixx_t& dtau_dv2, matrixx_t& dtau_df);

        matrix43_t QuatLinearization(int node);

        void SwingHeightLinearization(int node, const std::string& frame, matrix6x_t& jacobian);

        void HolonomicLinearizationq(int node, const std::string& frame, matrix6x_t& jacobian);
        void HolonomicLinearizationv(int node, const std::string& frame, matrix6x_t& jacobian);
    // ----------- Cost Creation ----------- //
        void CreateCostPattern();
        void UpdateCost();
        double GetFullCost(const vectorx_t& qp_res);
        // void CreateDefaultCost();
        // Helper function
        // void FormCostFcnArg(const vectorx_t& delta, const vectorx_t& bar, const vectorx_t& target, vectorx_t& arg) const;

    // ----- Sparsity Pattern Creation ----- //
        /**
         * @brief Create the constraint sparsity pattern.
         *
         * This will allocate the memory for all the triplets by filling them with dummy values.
         * Then this will fill the A matrix.
         * Then this will reset the triplet index
         */
        void CreateConstraintSparsityPattern();
        void AddICPattern();
        void AddIntegrationPattern(int node);
        void AddIDPattern(int node);
        void AddFrictionConePattern(int node);
        void AddConfigurationBoxPattern(int node);
        void AddVelocityBoxPattern(int node);
        void AddTorqueBoxPattern(int node);
        void AddSwingHeightPattern(int node);
        void AddHolonomicPattern(int node);

    // ----- Helper Functions ----- //
        void ConvertSolutionToTraj(const vectorx_t& qp_sol, Trajectory& traj);

        [[nodiscard]] int GetNumConstraints() const;
        [[nodiscard]] int GetConstraintsPerNode() const;
        [[nodiscard]] int GetNumDecisionVars() const;
        int GetDecisionVarsPerNode() const;
        int GetDecisionIdx(int node, const DecisionType& var_type) const;
        int GetConstraintRow(int node, const ConstraintType& constraint) const;

        void MatrixToNewTriplet(const matrixx_t& mat, int row_start, int col_start, std::vector<Eigen::Triplet<double>>& triplet);
        void VectorToNewTriplet(const vectorx_t& vec, int row_start, int col_start, std::vector<Eigen::Triplet<double>>& triplet);
        void MatrixToTriplet(const matrixx_t& mat, int row_start, int col_start, std::vector<Eigen::Triplet<double>>& triplet, int& triplet_idx, bool prune_zeros=false);
        void VectorToTriplet(const vectorx_t& vec, int row_start, int col_start, std::vector<Eigen::Triplet<double>>& triplet, int& triplet_idx);
        void DiagonalMatrixToTriplet(const matrixx_t& mat, int row_start, int col_start, std::vector<Eigen::Triplet<double>>& triplet, int& triplet_idx);
        /**
         * @brief Assign a matrix that is diagonal and all of the same value to triplets. Assumes the matrix is square.
         * @param val
         * @param row_start
         * @param col_start
         * @param size
         * @param triplet
         * @param triplet_idx
         */
        void DiagonalScalarMatrixToTriplet(double val, int row_start, int col_start, int size, std::vector<Eigen::Triplet<double>>& triplet, int& triplet_idx);

    // ----- Getters for Sizes of Individual nodes ----- //
        [[nodiscard]] int NumIntegratorConstraintsNode() const;
        [[nodiscard]] int NumIDConstraintsNode() const;
        [[nodiscard]] int NumFrictionConeConstraintsNode() const;
        [[nodiscard]] int NumConfigBoxConstraintsNode() const;
        [[nodiscard]] int NumVelocityBoxConstraintsNode() const;
        [[nodiscard]] int NumTorqueBoxConstraintsNode() const;
        [[nodiscard]] int NumSwingHeightConstraintsNode() const;
        [[nodiscard]] int NumHolonomicConstraintsNode() const;

        void UpdateSettings();

        static constexpr int CONTACT_3DOF = 3;
        static constexpr int FLOATING_VEL = 6;
        static constexpr int FLOATING_BASE = 7;
        static constexpr int FRICTION_CONE_SIZE = 4;
        static constexpr int POS_VARS = 3;
        static constexpr int QUAT_VARS = 4;
        static constexpr double FD_DELTA = 1e-8;

    //---------- Member Variables ---------- //
        fs::path config_file_;

        // OSQP Interface
        osqp::OsqpInstance osqp_instance_;
        osqp::OsqpSolver osqp_solver_;
        osqp::OsqpSettings osqp_settings_;

        sp_matrixx_t A_;
//        constraints::SparseBoxConstraints constraints_;

        // Hold the constraint matrix as a vector of triplets
        std::vector<Eigen::Triplet<double>> constraint_triplets_;
        int constraint_triplet_idx_{};

        // Codegen
        bool compile_derivatves_;

        // Cost
        CostFunction cost_;

        vectorx_t vel_tracking_weight_;
        vectorx_t config_tracking_weight_;

        std::vector<Eigen::Triplet<double>> objective_triplets_;
        int objective_triplet_idx_{};

        sp_matrixx_t objective_mat_;
        // vectorx_t objective_vec_;

        std::vector<vectorx_t> q_target_;
        std::vector<vectorx_t> v_target_;

        // Model
        std::unique_ptr<models::FullOrderRigidBody> robot_model_;

        // Warm start trajectory
        Trajectory traj_;

        // dt's
        std::vector<double> dt_;

        // Workspace
        std::unique_ptr<Workspace> ws_;

        // Recording state
        std::vector<MpcStats> stats_;

        // General settings
        bool verbose_;

        // Constraint settings
        double friction_coef_{};
        double max_grf_{};
        int nodes_{};

        // Contact settings
        int num_contact_locations_{};
        std::vector<std::string> contact_frames_{};

        // TODO: Populate
        std::map<std::string, std::vector<double>> swing_traj_;

    private:
    };
} // namepsace torc::mpc

#endif //TORC_FULL_ORDER_MPC_H
