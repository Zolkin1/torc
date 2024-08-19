//
// Created by zolkin on 7/24/24.
//

#include "full_order_mpc.h"

int main() {
 using namespace torc::mpc;
    std::filesystem::path achilles_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/achilles.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/achilles_mpc_config_sim.yaml";

    FullOrderMpc mpc("achilles_mpc", mpc_config, achilles_urdf);

    mpc.Configure();

    torc::models::FullOrderRigidBody achilles("achilles", achilles_urdf);

    torc::mpc::ContactSchedule cs(mpc.GetContactFrames());
    // cs.InsertContact("right_hand", 0, 1);
    // cs.InsertContact("left_hand", 0, 1);

    // cs.InsertContact("right_foot", 0, 1);
    // cs.InsertContact("foot_front_right", 0, 1);
    // cs.InsertContact("foot_rear_right", 0, 1);

    cs.InsertContact("foot_front_right", 0.6, 1000);
    cs.InsertContact("foot_rear_right", 0.6, 1000);

    cs.InsertContact("foot_front_right", 0, 0.3);
    cs.InsertContact("foot_rear_right", 0, 0.3);

    // cs.InsertContact("left_foot", 0, 1);
    cs.InsertContact("foot_front_left", 0, 10);
    cs.InsertContact("foot_rear_left", 0, 10);
    mpc.UpdateContactScheduleAndSwingTraj(cs, 0.08, 0.01, 0.5);

    vectorx_t q_target, v_target;
    q_target.resize(achilles.GetConfigDim());
    q_target << 0., 0, 0.97,    // position
                0, 0, 0, 1,     // quaternion
                0, 0, -0.26,    // L hips joints
                0.65, -0.43,    // L knee, ankle
                0, 0, 0, 0,     // L shoulders and elbow
                0, 0, -0.26,    // R hip joints
                0.65, -0.43,    // R knee ankle
                0, 0, 0, 0;     // R shoulders and elbow

    mpc.SetConstantConfigTarget(q_target);
    q_target(0) = 0;

    v_target.resize(achilles.GetVelDim());
    v_target << 0, 0, 0,
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

    traj.SetDtVector(mpc.GetDtVector());

    std::cout << "robot mass: " << achilles.GetMass() << std::endl;
    double time = 0;
    for (int i = 0; i < traj.GetNumNodes(); i++) {
        int num_contacts = 0;
        for (const auto& frame : mpc.GetContactFrames()) {
            if (cs.InContact(frame, time)) {
                num_contacts++;
            }
        }

        for (const auto& frame : mpc.GetContactFrames()) {
            if (cs.InContact(frame, time)) {
                traj.SetForce(i, frame, {0, 0, 9.81*achilles.GetMass()/num_contacts});
            }
        }

        time += traj.GetDtVec()[i];
    }

    mpc.SetWarmStartTrajectory(traj);

    // std::cout << "press any key to start mpc... " << std::endl;
    // std::cin.get();

    mpc.ComputeNLP(q_target, v_target, traj);

    for (int i = 0; i < 40; i++) {
        vectorx_t q_current;
        traj.GetConfigInterp(0.01, q_current);
        std::cout << "q: " << q_current.transpose() << std::endl;
        vectorx_t v_current;
        traj.GetVelocityInterp(0.01, v_current);

        mpc.Compute(q_current, v_current, traj);
    }



    for (int i = 0; i < traj.GetNumNodes(); i++) {
        std::cout << "Node: " << i << std::endl;
        std::cout << "config: " << traj.GetConfiguration(i).transpose() << std::endl;
        std::cout << "vel: " << traj.GetVelocity(i).transpose() << std::endl;
        // std::cout << "torque: " << traj.GetTau(i).transpose() << std::endl;
        // achilles.SecondOrderFK(traj.GetConfiguration(i), traj.GetVelocity(i));
        // for (const auto& frame : mpc.GetContactFrames()) {
        //     std::cout << "frame: " << frame << "\npos: " << achilles.GetFrameState(frame).placement.translation().transpose() << std::endl;
        //     std::cout << "vel: " << achilles.GetFrameState(frame).vel.linear().transpose() << std::endl;
        //     std::cout << "force: " << traj.GetForce(i, frame).transpose() << std::endl;
        // }
        std::cout << std::endl;
    }

    // for (const auto& frame : mpc.GetContactFrames()) {
    //     std::cout << "frame: " << std::endl;
    //     for (int i = 0; i < traj.GetNumNodes(); i++) {
    //         std::cout << "node: " << i << ", z: " << mpc.swing_traj_[frame][i] << std::endl;
    //     }
    // }

    mpc.PrintStatistics();
    std::cout << std::endl << std::endl;
    mpc.PrintContactSchedule();
    std::cout << std::endl << std::endl;
}