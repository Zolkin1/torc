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

namespace torc::mpc {
    namespace fs = std::filesystem;

    using vectorx_t = Eigen::VectorXd;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3x_t = Eigen::Matrix3Xd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix43_t = Eigen::Matrix<double, 4, 3>;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    using sp_matrixx_t = Eigen::SparseMatrix<double>;

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
        Trajectory Compute(const vectorx_t& state);

        void SetVerbosity(bool verbose);


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

        // TODO: Consider adding a workspace where I can put intermediate values without needing to allocate memory
        struct Workspace {
            matrixx_t int_mat;
            matrixx_t id_state_mat;
            matrixx_t id_force_mat;
            matrixx_t fric_cone_mat;
            vectorx_t swing_vec;
            matrix6x_t frame_jacobian;
            matrixx_t holo_mat;
            vectorx_t acc;
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
        matrix3_t GetQuatIntegrationLinearizationXi(int node);
        matrix3_t GetQuatIntegrationLinearizationW(int node);

        matrix43_t GetQuatLinearization(int node);

    // ----------- Cost Creation ----------- //
        void CreateCost();

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
        [[nodiscard]] int GetNumConstraints() const;
        [[nodiscard]] int GetConstraintsPerNode() const;
        [[nodiscard]] int GetNumDecisionVars() const;
        int GetDecisionVarsPerNode() const;
        int GetDecisionIdx(int node, const DecisionType& var_type) const;
        int GetConstraintRow(int node, const ConstraintType& constraint) const;

        void MatrixToNewTriplet(const matrixx_t& mat, int row_start, int col_start);
        void VectorToNewTriplet(const vectorx_t& vec, int row_start, int col_start);
        void MatrixToTriplet(const matrixx_t& mat, int row_start, int col_start);
        void VectorToTriplet(const vectorx_t& vec, int row_start, int col_start);
        void DiagonalMatrixToTriplet(const matrixx_t& mat, int row_start, int col_start);
        /**
         * @brief Assign a matrix that is diagonal and all of the same value to triplets. Assumes the matrix is square.
         * @param val
         * @param row_start
         * @param col_start
         * @param size
         */
        void DiagonalScalarMatrixToTriplet(double val, int row_start, int col_start, int size);

    // ----- Getters for Sizes of Individual nodes ----- //
        [[nodiscard]] int NumIntegratorConstraintsNode() const;
        [[nodiscard]] int NumIDConstraintsNode() const;
        [[nodiscard]] int NumFrictionConeConstraintsNode() const;
        [[nodiscard]] int NumConfigBoxConstraintsNode() const;
        [[nodiscard]] int NumVelocityBoxConstraintsNode() const;
        [[nodiscard]] int NumTorqueBoxConstraintsNode() const;
        [[nodiscard]] int NumSwingHeightConstraintsNode() const;
        [[nodiscard]] int NumHolonomicConstraintsNode() const;

        void UpdateConfigurations();

        static constexpr int CONTACT_3DOF = 3;
        static constexpr int FLOATING_VEL = 6;
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

        sp_matrixx_t A;
//        constraints::SparseBoxConstraints constraints_;

        // Hold the constraint matrix as a vector of triplets
        std::vector<Eigen::Triplet<double>> constraint_triplets_;
        int triplet_idx_{};

        // Model
        std::unique_ptr<models::FullOrderRigidBody> robot_model_;

        // Warm start trajectory
        Trajectory traj_;

        // dt's
        std::vector<double> dt_;

        // Workspace
        std::unique_ptr<Workspace> ws_;

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
