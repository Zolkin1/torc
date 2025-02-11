//
// Created by zolkin on 1/19/25.
//

#include "pinocchio_interface.h"

#include "DynamicsConstraintsTest.h"

namespace torc::mpc {
    DynamicsConstraintsTest::DynamicsConstraintsTest(const models::FullOrderRigidBody &model,
        const std::vector<std::string> &contact_frames, const std::string &name, const fs::path &deriv_lib_path,
        bool compile_derivs, bool full_order, int first_node, int last_node)
            : DynamicsConstraint(model, contact_frames, name, deriv_lib_path, compile_derivs, full_order, first_node,
                last_node), contact_frames_(contact_frames) {}

    bool DynamicsConstraintsTest::ForwardDynamicsTest(const vectorx_t &q_lin, const vectorx_t &v1_lin,
        const vectorx_t &v2_lin, const vectorx_t &tau_lin, const vectorx_t &F_lin) {
        pinocchio::Model pin_model = model_.GetModel();
        pinocchio::Data pin_data(pin_model);

        std::vector<models::ExternalForce<double>> f_ext;
        for (int i = 0; i < contact_frames_.size(); i++) {
            f_ext.emplace_back(contact_frames_[i], F_lin.segment<3>(i*3));
        }

        // Compute the forward dynamics using aba on the differential
        vectorx_t a = models::ForwardDynamics<double>(pin_model, pin_data, q_lin, v1_lin, tau_lin, f_ext);



        // Compute the forward dynamics using the ID method here
    }

}