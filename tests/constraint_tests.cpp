#define CATCH_CONFIG_MAIN

#include <catch2/catch_test_macros.hpp>

#include "constraint.h"
#include "explicit_fn.h"
#include "test_fn.h"
#include "catch2/benchmark/catch_benchmark.hpp"

TEST_CASE("Constraint Initialization and Check", "[constraint]") {
    using namespace torc;
    auto constraint0 = constraint::Constraint<double>();
    const auto func_d = test::functions<double>;
    const auto grad_d = test::gradients<double>;
    const auto hess_d = test::hessians<double>;
    for (int i=0; i<func_d.size(); i++) {
        constraint0.AddConstraint(
            fn::ExplicitFn(func_d.at(i),
                           grad_d.at(i),
                           hess_d.at(i)),
            0.5,
            constraint::GreaterThan
        );
    }
    const Eigen::Vector3d vec0 = {1, 2, 3};
    REQUIRE_FALSE(constraint0.Check(vec0));
    REQUIRE_FALSE(constraint0.Check(Eigen::Vector3d::Zero()));

    auto constraint1 = constraint::Constraint<double>();
    constraint1.AddConstraint(
        fn::ExplicitFn(func_d.front(),
                        grad_d.front(),
                        hess_d.front()
            ),
            1,
            constraint::Equals
    );
    REQUIRE_FALSE(constraint1.Check(vec0));
    const Eigen::Vector2d vec1 = {1, 0};
    REQUIRE(constraint1.Check(vec1));
}


TEST_CASE("Constraint Forms", "[constraint]") {
    using namespace torc::constraint;
    using namespace torc::fn;
    auto constraint0 = Constraint<double>();
    const auto func_d = test::functions<double>;    // prepare to instantiate functions
    const auto grad_d = test::gradients<double>;
    const auto hess_d = test::hessians<double>;
    constraint0.AddConstraint(
        ExplicitFn(func_d.at(0),
                   grad_d.at(0),
                   hess_d.at(0)),
        -1,
        GreaterThan
    );
    constraint0.AddConstraint(
        ExplicitFn(func_d.at(1),
                   grad_d.at(1),
                   hess_d.at(1)),
        1,
        GreaterThan
    );
    constraint0.AddConstraint(
        ExplicitFn(func_d.at(2),
                   grad_d.at(2),
                   hess_d.at(2)),
        0,
        LessThan
    );
    constraint0.AddConstraint(
        ExplicitFn(func_d.at(3),
                   grad_d.at(3),
                   hess_d.at(3)),
        1,
        Equals
    );

    const Eigen::Vector3d vec = {1, 2, 3};      // vector to test
    ConstraintData<double> constraint_data;

    SECTION("Raw Form") {
        constraint0.OriginalForm(vec, constraint_data);
        Eigen::MatrixXd A_true(4, 3);
        A_true << 1, 1, 1,
                  2, 4, 6,
                  108, 108, 108,
                  6, 3, 2;
        REQUIRE(constraint_data.ineq_grad == A_true);
        Eigen::VectorXd bounds_true(4);
        bounds_true << -7, -13, -216, -5;
        REQUIRE(constraint_data.bound_all == bounds_true);
        std::vector types_true = {GreaterThan, GreaterThan, LessThan, Equals};
        REQUIRE(constraint_data.types == types_true);
    }

    SECTION("Compact Raw Form") {
        constraint0.CompactOriginalForm(vec, constraint_data);
        Eigen::MatrixXd A_true(4, 3);
        A_true << 1, 1, 1,
                  2, 4, 6,
                  108, 108, 108,
                  6, 3, 2;
        REQUIRE(constraint_data.ineq_grad == A_true);
        Eigen::VectorXd bounds_true(4);
        bounds_true << -7, -13, -216, -5;
        REQUIRE(constraint_data.bound_all == bounds_true);
        std::vector types_true = {GreaterThan, LessThan, Equals};
        REQUIRE(constraint_data.types == types_true);
        std::vector<size_t> annotations_true {2, 1, 1};
        REQUIRE(constraint_data.reps == annotations_true);
    }

    SECTION("Unilateral Form") {
        constraint0.UnilateralForm(vec, constraint_data);
        Eigen::MatrixXd A_true(5, 3);
        A_true << -1, -1, -1,
                  -2, -4, -6,
                  108, 108, 108,
                  6, 3, 2,
                  -6, -3, -2;
        REQUIRE(constraint_data.ineq_grad == A_true);
        Eigen::VectorXd bounds_true(5);
        bounds_true << 7, 13, -216, -5, 5;
        REQUIRE(constraint_data.bound_high == bounds_true);

        constraint0.SparseUnilateralForm(vec, constraint_data);
        REQUIRE(A_true == constraint_data.ineq_grad_sparse.toDense());
        REQUIRE(constraint_data.bound_high == bounds_true);
    }

    SECTION("Box Form") {
        constraint0.BoxForm(vec, constraint_data);
        Eigen::MatrixXd A_true(4, 3);
        A_true << 1, 1, 1,
                  2, 4, 6,
                  108, 108, 108,
                  6, 3, 2;
        REQUIRE(constraint_data.ineq_grad == A_true);
        Eigen::VectorXd lbounds_true(4), ubounds_true(4);
        constexpr double max = std::numeric_limits<double>::max();
        constexpr double min = -max;
        lbounds_true << -7, -13, min, -5;
        ubounds_true << max, max, -216, -5;
        REQUIRE(constraint_data.bound_low == lbounds_true);
        REQUIRE(constraint_data.bound_high == ubounds_true);

        constraint0.SparseBoxForm(vec, constraint_data);
        REQUIRE(A_true == constraint_data.ineq_grad_sparse.toDense());
        REQUIRE(constraint_data.bound_low == lbounds_true);
        REQUIRE(constraint_data.bound_high == ubounds_true);
    }

    SECTION("Inequality Equality Form") {
        constraint0.InequalityEqualityForm(vec, constraint_data);
        Eigen::MatrixXd A_true(3, 3);
        Eigen::MatrixXd G_true(1, 3);
        A_true << -1, -1, -1,
                  -2, -4, -6,
                  108, 108, 108;
        G_true << 6, 3, 2;
        REQUIRE(constraint_data.ineq_grad== A_true);
        REQUIRE(constraint_data.eq_grad== G_true);
        Eigen::VectorXd bounds0_true(3), bounds1_true(1);
        bounds0_true << 7, 13, -216;
        bounds1_true << -5;
        REQUIRE(constraint_data.bound_high == bounds0_true);
        REQUIRE(constraint_data.bound_eq == bounds1_true);

        constraint0.SparseInequalityEqualityForm(vec, constraint_data);
        REQUIRE(A_true == constraint_data.ineq_grad_sparse.toDense());
        REQUIRE(G_true == constraint_data.eq_grad_sparse.toDense());
        REQUIRE(constraint_data.bound_high == bounds0_true);
        REQUIRE(constraint_data.bound_eq == bounds1_true);
    }
}
