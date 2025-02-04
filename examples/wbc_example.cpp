//
// Created by zolkin on 2/3/25.
//

#include <torc_timer.h>

#include "MpcSettings.h"
#include "wbc_controller.h"

int main() {
    // Create all the constraints
    using namespace torc::mpc;
    std::filesystem::path g1_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_hand.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_mpc_config.yaml";

    MpcSettings settings(mpc_config);
    settings.Print();

    const std::string pin_model_name = "test_pin_model";

    torc::models::FullOrderRigidBody g1("g1", g1_urdf, settings.joint_skip_names, settings.joint_skip_values);

    fs::path deriv_lib_path = fs::current_path();
    deriv_lib_path = deriv_lib_path / "deriv_libs";

    vectorx_t base_weight, joint_weight, tau_weight, force_weight, kp, kd;
    base_weight = vectorx_t::Constant(6, 10);
    joint_weight = vectorx_t::Ones(g1.GetVelDim() - 6) * 0.1;
    tau_weight = joint_weight;
    force_weight = vectorx_t::Constant(12, 0.001);
    kp = vectorx_t::Constant(g1.GetVelDim(), 10);
    kd = vectorx_t::Constant(g1.GetVelDim(), 1);

    torc::controller::WbcController controller(g1, settings.contact_frames,
        base_weight, joint_weight, tau_weight, force_weight, kp, kd,
        settings.friction_coef, true,
        settings.deriv_lib_path, settings.compile_derivs);

    std::vector<bool> in_contact = {true, true, true, true};

    torc::utils::TORCTimer timer;

    timer.Tic();
    vectorx_t tau = controller.ComputeControl(g1.GetNeutralConfig(), vectorx_t::Zero(g1.GetVelDim()),
        g1.GetNeutralConfig(), vectorx_t::Zero(g1.GetVelDim()), vectorx_t::Zero(g1.GetVelDim() - 6),
        vectorx_t::Constant(12, 10), in_contact);
    timer.Toc();
    std::cout << "Entire compute took " << timer.Duration<std::chrono::microseconds>().count()*1e-3 << " ms" << std::endl;


    std::cout << "tau: " << tau.transpose() << std::endl;
}