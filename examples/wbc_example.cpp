//
// Created by zolkin on 2/3/25.
//

#include <torc_timer.h>

#include "MpcSettings.h"
#include "wbc_controller.h"

int main() {
    // Create all the constraints
    using namespace torc::mpc;
    std::filesystem::path g1_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_hand_v2.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_mpc_config.yaml";

    MpcSettings settings(mpc_config);
    settings.Print();

    const std::string pin_model_name = "test_pin_model";


    torc::controller::WbcSettings wbc_settings(mpc_config);

    torc::models::FullOrderRigidBody g1("g1", g1_urdf, wbc_settings.skip_joints, wbc_settings.joint_values);

    torc::controller::WbcController controller(g1, wbc_settings.contact_frames,
        wbc_settings,
        settings.friction_coef, true,
        settings.deriv_lib_path, wbc_settings.compile_derivs);

    std::vector<bool> in_contact = {true, true, true, true};

    torc::utils::TORCTimer timer;

    timer.Tic();
    vectorx_t tau = controller.ComputeControl(g1.GetRandomConfig(), vectorx_t::Zero(g1.GetVelDim()),
        g1.GetRandomConfig(), vectorx_t::Zero(g1.GetVelDim()), vectorx_t::Zero(g1.GetVelDim() - 6),
        vectorx_t::Constant(12, 10), in_contact);
    timer.Toc();
    std::cout << "Entire compute took " << timer.Duration<std::chrono::microseconds>().count()*1e-3 << " ms" << std::endl;

    std::cout << "tau: " << tau.transpose() << std::endl;

    for (const auto& j : g1.GetModel().names) {
        std::cout << j << std::endl;
    }
}