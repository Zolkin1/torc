#define CATCH_CONFIG_MAIN

#include <catch2/catch_test_macros.hpp>

#include "constraint.h"
#include "explicit_fn.h"
#include "test_fn.h"

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
            constraint::GEQ
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
            constraint::EQ
    );
    REQUIRE_FALSE(constraint1.Check(vec0));
    const Eigen::Vector2d vec1 = {1, 0};
    REQUIRE(constraint1.Check(vec1));
}


TEST_CASE("Constraint Forms", "[constraint]")