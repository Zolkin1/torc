//
// Created by zolkin on 3/13/25.
//

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "step_planner_tests.h"

TEST_CASE("Polytope Area Tester") {
    using namespace torc;
    std::vector<mpc::ContactInfo> contact_polys;
    contact_polys.push_back(mpc::ContactSchedule::GetDefaultContactInfo());
    contact_polys.push_back(mpc::ContactSchedule::GetDefaultContactInfo());
    contact_polys.push_back(mpc::ContactSchedule::GetDefaultContactInfo());
    contact_polys.push_back(mpc::ContactSchedule::GetDefaultContactInfo());

    std::vector<std::string> contact_frames;
    contact_frames.push_back("1");
    contact_frames.push_back("2");
    contact_frames.push_back("3");
    contact_frames.push_back("4");

    std::vector<double> contact_offsets;
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);

    step_planning::StepPlanTester planner(contact_polys, contact_frames, contact_offsets);

    SECTION("Basic Polytope Area") {
        planner.CheckPolytopeArea();
    }

    SECTION("Polytope and Circle Area") {
        planner.CheckPolytopeCircleArea();
    }
}

TEST_CASE("Sampling Tree Tester") {
    using namespace torc;
    std::vector<mpc::ContactInfo> contact_polys;
    mpc::ContactInfo p1, p2, p3, p4;
    p1.A_.resize(2,2);
    p2.A_.resize(2,2);
    p3.A_.resize(2,2);
    p4.A_.resize(2,2);

    p1.A_ << 1, 0, 0, 1;
    p1.b_ << 1, 1, 0.125, 0.125;
    p2.A_ = p1.A_;
    p2.b_ << -0.125, 1, -1, 0.125;
    p3.A_ = p1.A_;
    p3.b_ << -0.125, -0.125, -1, -1;
    p4.A_ = p1.A_;
    p4.b_ << 1, -0.125, 0.125, -1;
    contact_polys.push_back(p1);
    contact_polys.push_back(p2);
    contact_polys.push_back(p3);
    contact_polys.push_back(p4);

    std::vector<std::string> contact_frames;
    contact_frames.push_back("1");
    contact_frames.push_back("2");
    contact_frames.push_back("3");
    contact_frames.push_back("4");

    std::vector<double> contact_offsets;
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);

    step_planning::StepPlanTester planner(contact_polys, contact_frames, contact_offsets);

    SECTION("Tree Creation") {
        planner.CheckSampleTreeCreation();
    }

    SECTION("Branch Pruning") {
        planner.CheckBranchPruning();
    }
}

TEST_CASE("Sampling Tester") {
    using namespace torc;
    std::vector<mpc::ContactInfo> contact_polys;
    mpc::ContactInfo p1, p2, p3, p4;
    p1.A_.resize(2,2);
    p2.A_.resize(2,2);
    p3.A_.resize(2,2);
    p4.A_.resize(2,2);

    p1.A_ << 1, 0, 0, 1;
    p1.b_ << 1, 1, 0.125, 0.125;
    p2.A_ = p1.A_;
    p2.b_ << -0.125, 1, -1, 0.125;
    p3.A_ = p1.A_;
    p3.b_ << -0.125, -0.125, -1, -1;
    p4.A_ = p1.A_;
    p4.b_ << 1, -0.125, 0.125, -1;
    contact_polys.push_back(p1);
    contact_polys.push_back(p2);
    contact_polys.push_back(p3);
    contact_polys.push_back(p4);

    std::vector<std::string> contact_frames;
    contact_frames.push_back("1");
    contact_frames.push_back("2");
    contact_frames.push_back("3");
    contact_frames.push_back("4");

    std::vector<double> contact_offsets;
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);
    contact_offsets.push_back(0);

    step_planning::StepPlanTester planner(contact_polys, contact_frames, contact_offsets);

    planner.CheckPolytopeSampling();
}