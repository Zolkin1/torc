//
// Created by zolkin on 7/24/24.
//

#include "full_order_mpc.h"

int main() {
    using namespace torc::models;
    using namespace torc::mpc;
    const std::string pin_model_name = "test_pin_model";

    std::filesystem::path a1_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/test_a1.urdf";
    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/mpc_config.yaml";

    FullOrderMpc mpc(mpc_config, a1_urdf);

    mpc.Configure();
}