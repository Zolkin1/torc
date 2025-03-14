//
// Created by zolkin on 3/13/25.
//

#ifndef STEP_PLANNER_TESTS_H
#define STEP_PLANNER_TESTS_H

#include <catch2/catch_test_macros.hpp>
#include "step_planner.h"

namespace torc::step_planning {
    class StepPlanTester : public StepPlanner {
        public:
        StepPlanTester(const std::vector<mpc::ContactInfo>& contact_polytopes, const std::vector<std::string>& contact_frames,
            const std::vector<double>& contact_offsets) : StepPlanner(contact_polytopes, contact_frames, contact_offsets, 0, 0, 0) {}

        void CheckPolytopeCircleArea() {
            // Make a polytope that has known area
            mpc::ContactInfo poly = mpc::ContactSchedule::GetDefaultContactInfo();

            // Make a circle that it is fully in
            vector2_t point = {101, 101};
            double rad = 10000;
            double computed_area = GetPolytopeCircleArea(poly, point, rad);
            CHECK(computed_area == 200*200);

            // Make a circle that it is fully out
            point = {-1000, -1000};
            rad = 1;
            computed_area = GetPolytopeCircleArea(poly, point, rad);
            CHECK(computed_area == 0);

            // Make a partial covering
            point = {100, 100};
            rad = 10;
            computed_area = GetPolytopeCircleArea(poly, point, rad);
            CHECK(std::abs(computed_area - 100.*0.78539816339) < 1e-3);

            // Make a partial covering
            point = {-100, -100};
            rad = 10;
            computed_area = GetPolytopeCircleArea(poly, point, rad);
            CHECK(std::abs(computed_area - 100.*0.78539816339) < 1e-3);

            // Make a partial covering
            point = {0, 100};
            rad = 10;
            computed_area = GetPolytopeCircleArea(poly, point, rad);
            CHECK(std::abs(computed_area - 100.*2.*0.78539816339) < 1e-3);

            // Make a partial covering
            point = {100, 0};
            rad = 10;
            computed_area = GetPolytopeCircleArea(poly, point, rad);
            CHECK(std::abs(computed_area - 100.*2.*0.78539816339) < 1e-3);

            mpc::ContactInfo p1;
            p1.A_.resize(2, 2);
            p1.A_ << 1, 0, 0, 1;
            p1.b_ << 1, 1, 0.125, 0.125;
            point = {0, 0};
            rad = 0.25;
            computed_area = GetPolytopeCircleArea(p1, point, rad);
            CHECK(computed_area > 0);
        }

        void CheckPolytopeArea() {
            // Make a polytope that has known area
            std::vector<vector2_t> points;
            points.push_back(vector2_t(0, 0));
            points.push_back(vector2_t(0, 1));
            points.push_back(vector2_t(1, 1));
            points.push_back(vector2_t(1, 0));

            // Call the function and check
            double computed_area = ComputePolytopeArea(points);
            CHECK(computed_area == 1.);

            points.clear();
            points.push_back(vector2_t(0, 0));
            points.push_back(vector2_t(0, 2));
            points.push_back(vector2_t(1, 2));
            points.push_back(vector2_t(1, 0));

            // Call the function and check
            computed_area = ComputePolytopeArea(points);
            CHECK(computed_area == 2.);

            points.clear();
            points.push_back(vector2_t(0.5, 0));
            points.push_back(vector2_t(0.5, 2));
            points.push_back(vector2_t(1.5, 2));
            points.push_back(vector2_t(1.5, 0));

            // Call the function and check
            computed_area = ComputePolytopeArea(points);
            CHECK(computed_area == 2.);

            points.clear();
            points.push_back(vector2_t(0, 0));
            points.push_back(vector2_t(0, 2));
            points.push_back(vector2_t(2, 0));

            // Call the function and check
            computed_area = ComputePolytopeArea(points);
            CHECK(computed_area == 2.);

            points.clear();
            points.push_back(vector2_t(100, 100));
            points.push_back(vector2_t(100, -100));
            points.push_back(vector2_t(-100, -100));
            points.push_back(vector2_t(-100, 100));

            // Call the function and check
            computed_area = ComputePolytopeArea(points);
            CHECK(computed_area == 200*200);
        }

        void CheckSampleTreeCreation() {
            std::vector<vector2_t> points;
            points.push_back(vector2_t(0, 0));
            points.push_back(vector2_t(0., 0));

            std::shared_ptr<SampleTreeNode> root = CreateSampleTree(points, 0.25);
            CHECK(root->GetNumChildren() == 4);
            for (int i = 0; i < root->GetNumChildren(); i++) {
                CHECK(root->children[i]->GetNumChildren() == 4);
                CHECK(root->children[i]->polytope_idx == i);
            }


            points.clear();
            points.push_back(vector2_t(0, 0));
            points.push_back(vector2_t(0., 0.15));

            root = CreateSampleTree(points, 0.25);
            CHECK(root->GetNumChildren() == 4);
            for (int i = 0; i < root->GetNumChildren(); i++) {
                CHECK(root->children[i]->GetNumChildren() == 2);
            }

            points.clear();
            points.push_back(vector2_t(0, 0));
            points.push_back(vector2_t(0.5, 0.5));

            CHECK_THROWS(CreateSampleTree(points, 0.25));

            points.clear();
            points.push_back(vector2_t(0.5, 0));
            points.push_back(vector2_t(-0.5, 0));

            root = CreateSampleTree(points, 0.25);
            CHECK(root->GetNumChildren() == 2);
            for (int i = 0; i < root->GetNumChildren(); i++) {
                CHECK(root->children[i]->GetNumChildren() == 2);
            }
        }

        void CheckBranchPruning() {
            std::vector<vector2_t> points;
            points.push_back(vector2_t(0, 0));
            points.push_back(vector2_t(0., 0));

            std::shared_ptr<SampleTreeNode> root = CreateSampleTree(points, 0.25);
            CHECK(root->GetNumChildren() == 4);
            for (int i = 0; i < root->GetNumChildren(); i++) {
                REQUIRE(root->children[i]);
                CHECK(root->children[i]->GetNumChildren() == 4);
            }

            std::vector<int> branch = {0, 0};
            root->PruneBranch(branch);
            CHECK(root->GetNumChildren() == 4);
            CHECK(root->children[0]->GetNumChildren() == 3);
            for (int i = 1; i < root->GetNumChildren(); i++) {
                CHECK(root->children[i]->GetNumChildren() == 4);
            }

            branch = {0, 1};
            root->PruneBranch(branch);
            CHECK(root->GetNumChildren() == 4);
            CHECK(root->children[0]->GetNumChildren() == 2);
            for (int i = 1; i < root->GetNumChildren(); i++) {
                CHECK(root->children[i]->GetNumChildren() == 4);
            }

            branch = {0, 2};
            root->PruneBranch(branch);
            CHECK(root->GetNumChildren() == 4);
            CHECK(root->children[0]->GetNumChildren() == 1);
            for (int i = 1; i < root->GetNumChildren(); i++) {
                CHECK(root->children[i]->GetNumChildren() == 4);
            }

            branch = {0, 3};
            root->PruneBranch(branch);
            CHECK(root->GetNumChildren() == 3);
            for (int i = 0; i < root->GetNumChildren(); i++) {
                CHECK(root->children[i]->GetNumChildren() == 4);
            }

            branch = {2,1};
            root->PruneBranch(branch);
            CHECK(root->GetNumChildren() == 3);
            CHECK(root->children[1]->GetNumChildren() == 3);
            for (int i = 0; i < root->GetNumChildren(); i++) {
                if (i != 1) {
                    CHECK(root->children[i]->GetNumChildren() == 4);
                }
            }
        }

        void CheckPolytopeSampling() {
            std::vector<vector2_t> points;
            points.push_back(vector2_t(0, 0));
            std::vector<std::vector<int>> used_polys;

            auto sampled = SamplePolytopes(points, used_polys);
            REQUIRE(sampled.size() == points.size());
            CHECK(sampled[0].second == 2);

            // TODO: Add more tests
        }
    };
}

#endif //STEP_PLANNER_TESTS_H
