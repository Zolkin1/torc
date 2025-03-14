//
// Created by zolkin on 2/12/25.
//

#ifndef STEP_PLANNER_H
#define STEP_PLANNER_H

#include <random>

#include "proxsuite/proxqp/dense/dense.hpp"

#include "contact_schedule.h"
#include "simple_trajectory.h"

namespace torc::step_planning {
    using vectorx_t = Eigen::VectorXd;
    using vector2_t = Eigen::Vector2d;
    using vector4_t = Eigen::Vector4d;
    using matrix2_t = Eigen::Matrix2d;
    using matrix3_t = Eigen::Matrix3d;
    using matrixx_t = Eigen::MatrixXd;
    using quat_t = Eigen::Quaterniond;

    struct SampleTreeNode : public std::enable_shared_from_this<SampleTreeNode> {
        int polytope_idx;
        double area;
        std::vector<std::shared_ptr<SampleTreeNode>> children;
        std::weak_ptr<SampleTreeNode> parent;

        SampleTreeNode(int polytope_idx, double area) {
            this->polytope_idx = polytope_idx;
            this->area = area;
        }

        void AddChild(int polytope_idx, double area) {
            children.push_back(std::make_shared<SampleTreeNode>(polytope_idx, area));
            children.back()->parent = shared_from_this();
        }

        int GetNumChildren() {
            return children.size();
        }

        void RemoveChild(int polytope_idx) {
            auto it = std::remove_if(children.begin(), children.end(),
            [&](const std::shared_ptr<SampleTreeNode>& child) {
                return child && child->polytope_idx == polytope_idx;
            });

            if (it != children.end()) {
                children.erase(it, children.end()); // Remove matching elements
                // std::cerr << "Removed a child!" << std::endl;
                return;
            }
            std::cerr << "Provided index: " << polytope_idx << std::endl;
            std::cerr << "possible indexes: ";
            for (int i = 0; i < children.size(); i++) {
                if (children[i]) {
                    std::cerr << "i: " << i << " idx: " << children[i]->polytope_idx << ", ";
                } else {
                    std::cerr << "i: " << i << " is a nullptr!" << ", ";
                }
            }
            throw std::runtime_error("[SampleTree] Could not find a child of SampleTreeNode");
        }

        void PruneBranch(std::vector<int> branch_idxs) {
            if (branch_idxs.size() > 1) {
                // Follow the tree path
                auto it = std::find_if(children.begin(), children.end(),
                [&](const std::shared_ptr<SampleTreeNode>& child) {
                    return child->polytope_idx == branch_idxs[0];
                });

                if (it != children.end()) {
                    int branch_child_idx = branch_idxs[0];
                    branch_idxs.erase(branch_idxs.begin()); // Remove the first element
                    children[it - children.begin()]->PruneBranch(branch_idxs);
                    if (children[it - children.begin()]->children.size() == 0) {
                        RemoveChild(children[it - children.begin()]->polytope_idx);
                    }
                } else {
                    std::cerr << "Looking for idx: " << branch_idxs[0] << std::endl;
                    throw std::runtime_error("[SampleTree] Invalid branch to prune!");
                }
            } else {
                if (branch_idxs.size() == 0) {
                    throw std::runtime_error("[SampleTree] Empty branch!");
                }

                RemoveChild(branch_idxs[0]);
            }
        }
    };

    class StepPlanner {
    public:
        StepPlanner(const std::vector<mpc::ContactInfo>& contact_polytopes, const std::vector<std::string>& contact_frames,
            const std::vector<double>& contact_offsets, double current_time_buffer, double polytope_buffer);

        StepPlanner(const std::vector<mpc::ContactInfo>& contact_polytopes, const std::vector<std::string>& contact_frames,
            const std::vector<double>& contact_offsets, double current_time_buffer, double polytope_buffer, int seed);
        /**
         * @brief Chooses the contact polytopes based on the raibert heuristic
         * @param q current state
         * @param v current velocity
         * @param q_target target configuration
         * @param v_target target velocity
         * @param contact_schedule [output] current contact schedule that will be updated with the new contact polytopes
         * @param nominal_footholds [output] nominal foothold based on raibert
         * @param projected_footholds [output] the foothold projected onto the polytope
         */
        void PlanStepsHeuristic(const mpc::SimpleTrajectory& q_target,
                                const std::vector<double>& dt_vec,
                                mpc::ContactSchedule& contact_schedule,
                                std::map<std::string, std::vector<vector2_t>>& nominal_footholds,
                                std::map<std::string, std::vector<vector2_t>>& projected_footholds,
                                bool first_loop = false);

        // TODO: I will need to adjust this based on how it will be used.
        //  specifically, I will need to be careful with thread safety.
        /**
         * @brief Chooses the contact polytopes based on the raibert heuristic
         * @param q current state
         * @param v current velocity
         * @param q_target target configuration
         * @param v_target target velocity
         * @param contact_schedule [output] vector of current contact schedules that will be updated with the new contact polytopes.
         *  One contact schedule for each parallel MPC
         * @param nominal_footholds [output] nominal foothold based on raibert
         * @param projected_footholds [output] the foothold projected onto the polytope
         */
        void PlanStepsSampling(const mpc::SimpleTrajectory& q_target,
                                const std::vector<double>& dt_vec,
                                std::vector<mpc::ContactSchedule>& contact_schedule,
                                std::map<std::string, std::vector<vector2_t>>& nominal_footholds,
                                std::map<std::string, std::vector<vector2_t>>& projected_footholds,
                                bool first_loop = false);

        void UpdateContactPolytopes(const std::vector<mpc::ContactInfo>& contact_polytopes);

    protected:

        std::vector<double> ComputeContactMidtimes(const std::string& frame,
            const mpc::ContactSchedule& contact_schedule, double traj_end_time);

        static vectorx_t InterpolateTarget(const mpc::SimpleTrajectory& target, const std::vector<double>& dt_vec, double time);

        static bool InPolytope(const mpc::ContactInfo& polytope, const vector2_t& point);

        void SetFootTargetAndPolytope(double midtime, int contact_idx,
            const mpc::SimpleTrajectory& q_target, const std::vector<double>& dt_vec, int frame_idx,
            mpc::ContactSchedule& contact_schedule,
            std::map<std::string, std::vector<vector2_t>>& nominal_footholds,
            std::map<std::string, std::vector<vector2_t>>& projected_footholds);

        std::pair<vector2_t, int> ProjectOntoClosestPolytope(const vector2_t& point);

        std::pair<vector2_t, double> ProjectOntoPolytope(const vector2_t& point, const mpc::ContactInfo& polytope);

        /**
         * @brief
         * @param points All the points that need to have polytopes sampled
         * @param used_polys All the polytope groups that cannot be sampled
         * @return The projected point and the index for the selected polytope
         */
        std::vector<std::pair<vector2_t, int>> SamplePolytopes(const std::vector<vector2_t>& points, const std::vector<std::vector<int>>& used_polys);

        void SetFootTargetAndPolytopeSampling(double midtime, int contact_idx,
            const mpc::SimpleTrajectory& q_target, const std::vector<double>& dt_vec,
            const std::vector<std::vector<int>>& used_polys,
            mpc::ContactSchedule& contact_schedule,
            std::map<std::string, std::vector<vector2_t>>& nominal_footholds,
            std::map<std::string, std::vector<vector2_t>>& projected_footholds);

        int GetPolytopeIdx(const mpc::ContactInfo& polytope);

        double GetPolytopeCircleArea(const mpc::ContactInfo& polytope, const vector2_t& point, double radius);

        std::shared_ptr<SampleTreeNode> CreateSampleTree(const std::vector<vector2_t>& points, double sample_rad);

        double ComputePolytopeArea(std::vector<vector2_t> points);

        bool CircleLineIntersection(const vector2_t& p1, const vector2_t& p2, const vector2_t& center, double rad,
            std::vector<vector2_t>& intersection);

        std::vector<mpc::ContactInfo> contact_polytopes_;
        std::vector<std::string> contact_frames_;
        std::vector<vector2_t> contact_offsets_;
        double current_time_buffer_;    // Only update the polytopes that start after the current time buffer
        double polytope_buffer_;

        // Information for the random numbers
        std::random_device rd_;
        std::mt19937 gen_;

        // ProxQpInterface
        proxsuite::proxqp::dense::QP<double> qp_;

    private:
    };
}



#endif //STEP_PLANNER_H
