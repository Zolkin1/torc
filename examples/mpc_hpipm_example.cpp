//
// Created by zolkin on 1/18/25.
//

#include "DynamicsConstraint.h"
#include "hpipm_mpc.h"

int main() {
    // Create all the constraints
    using namespace torc::mpc;
    std::filesystem::path g1_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_hand.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_mpc_config.yaml";

    MpcSettings settings(mpc_config);
    settings.Print();

    const std::string pin_model_name = "test_pin_model";

    torc::models::FullOrderRigidBody g1("g1", g1_urdf);

    fs::path deriv_lib_path = fs::current_path();
    deriv_lib_path = deriv_lib_path / "deriv_libs";

    std::vector<std::string> contact_frames = {"left_toe", "left_heel", "right_toe", "right_heel"};
    std::vector<DynamicsConstraint> dynamics_constraints;
    dynamics_constraints.emplace_back(DynamicsConstraint(g1, contact_frames, "g1_full_order",
        deriv_lib_path, settings.compile_derivs, true, 0, 5));
    dynamics_constraints.emplace_back(g1, contact_frames, "g1_centroidal", deriv_lib_path,
        settings.compile_derivs, false, 5, 32);

    std::cout << "===== Constraints Created =====" << std::endl;

    HpipmMpc mpc(settings);
    mpc.SetDynamicsConstraints(std::move(dynamics_constraints));
}
