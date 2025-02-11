//
// Created by zolkin on 1/27/25.
//

#ifndef COLLISIONCONSTRAINT_H
#define COLLISIONCONSTRAINT_H
#include "constraint.h"
#include "full_order_rigid_body.h"
#include "CollisionData.h"

namespace torc::mpc {
    class CollisionConstraint : public Constraint {
    public:
        CollisionConstraint(int first_node, int last_node, const std::string& name,
            const models::FullOrderRigidBody& model, const std::filesystem::path& deriv_lib_path, bool compile_derivs,
            const std::vector<CollisionData>& collision_data);

        std::pair<matrixx_t, vectorx_t> GetLinearization(const vectorx_t& q_lin, int collision_idx);

        int GetNumConstraints() const override;

        int GetNumCollisions() const;

        vectorx_t GetViolation(const vectorx_t& q, const vectorx_t& dq, int collision_idx);

    protected:
    private:
        void CollisionFunction(const std::string& frame1, const std::string& frame2,
        const ad::ad_vector_t& dqk, const ad::ad_vector_t& qk_r1_r2, ad::ad_vector_t& violation);

        int config_dim_;

        std::vector<CollisionData> collision_data_;
        models::FullOrderRigidBody model_;
        std::vector<std::unique_ptr<ad::CppADInterface>> collision_function_;
    };
}


#endif //COLLISIONCONSTRAINT_H
