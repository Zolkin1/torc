//
// Created by zolkin on 7/24/24.
//

#include "full_order_mpc.h"

int main() {
 using namespace torc::mpc;
    std::filesystem::path achilles_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/achilles.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/achilles_mpc_config.yaml";

    FullOrderMpc mpc("achilles_mpc", mpc_config, achilles_urdf);

    mpc.Configure();

    torc::models::FullOrderRigidBody achilles("achilles", achilles_urdf);

    torc::mpc::ContactSchedule cs(mpc.GetContactFrames());
    cs.InsertContact("right_foot", 0, 1);
    // cs.InsertContact("left_foot", 0, 1);
    mpc.UpdateContactSchedule(cs);

    mpc.UpdateContactSchedule(cs);

    vectorx_t q_target, v_target;
    q_target.resize(achilles.GetConfigDim());
    q_target << 1, 0, 0.97,
                0, 0, 0, 1,
                // 0.7071, 0, 0.7071, 0,
                0, 0, -0.26,
                0, 0.65, -0.43,
                0, 0, 0,
                0, 0, -0.26,
                0.65, -0.43,
                0, 0, 0;
    mpc.SetConstantConfigTarget(q_target);
    q_target(0) = 0;

    v_target.resize(achilles.GetVelDim());
    v_target << 1, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0;
    mpc.SetConstantVelTarget(v_target);

    Trajectory traj;
    traj.UpdateSizes(achilles.GetConfigDim(), achilles.GetVelDim(), achilles.GetNumInputs(), mpc.GetContactFrames(), mpc.GetNumNodes());
    traj.SetDefault(q_target);

    mpc.SetWarmStartTrajectory(traj);

    mpc.Compute(q_target, v_target, traj);

    for (int i = 0; i < 2; i++) {
        mpc.Compute(q_target, v_target, traj);
    }

    for (int i = 0; i < traj.GetNumNodes(); i++) {
        std::cout << "Node: " << i << std::endl;
        std::cout << "config: " << traj.GetConfiguration(i).transpose() << std::endl;
        std::cout << "vel: " << traj.GetVelocity(i).transpose() << std::endl;
        std::cout << "torque: " << traj.GetTau(i).transpose() << std::endl;
        for (const auto& frame : mpc.GetContactFrames()) {
            std::cout << "frame: " << frame << ", force: " << traj.GetForce(i, frame).transpose() << std::endl;
        }
        std::cout << std::endl;
    }

    mpc.PrintStatistics();
    std::cout << std::endl << std::endl;
    mpc.PrintContactSchedule();
    std::cout << std::endl << std::endl;
}