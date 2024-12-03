//
// Created by zolkin on 10/21/24.
//

#include "full_order_mpc.h"


int main() {
    using namespace torc::mpc;
    std::filesystem::path achilles_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_hand.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_mpc_config.yaml";

    FullOrderMpc mpc("g1_mpc", mpc_config, achilles_urdf);

    mpc.Configure();

    torc::models::FullOrderRigidBody g1("g1", achilles_urdf, mpc.GetJointSkipNames(), mpc.GetJointSkipValues());

    torc::mpc::ContactSchedule cs(mpc.GetContactFrames());

    cs.InsertSwing("right_toe", 0.3, 0.6);
    cs.InsertSwing("right_heel", 0.3, 0.6);

    matrixx_t A_temp = matrixx_t::Identity(2, 2);
    Eigen::Vector4d b_temp = Eigen::Vector4d::Zero();
    b_temp << 10, 10, -10, -10;
    for (const auto& frame : mpc.GetContactFrames()) {
        for (int i = 0; i < cs.GetNumContacts(frame); i++) {
            cs.SetPolytope(frame, i, A_temp, b_temp);
        }
    }

    const double apex_height = 0.08;
    const std::vector<double> foot_height = {0.01, 0.01, 0.01, 0.01};
    const double apex_time = 0.75;
    mpc.UpdateContactScheduleAndSwingTraj(cs, apex_height, foot_height, apex_time);
    for (const auto& frame : mpc.GetContactFrames()) {
        mpc.PrintSwingTraj(frame);
    }

    // Print swing trajectory to csv for checking
    // std::ofstream traj_file;
    // traj_file.open("swing_traj.txt");
    // double t = 0.2;
    // for (int i = 0; i < 100; i++) {
    //     double height = cs.GetSwingHeight(apex_height, foot_height[0], apex_time, t, 0.2, 0.5);
    //     t += 0.3/100;
    //     traj_file << height << ", " << t << std::endl;
    // }
    std::cout << "Apex time: " << apex_time << " foot height: " << foot_height[0] << " start time: " << 0 << " end time: " << 0.3 << std::endl;
    // traj_file.close();

    vectorx_t q_target, v_target;
    q_target.resize(g1.GetConfigDim());
    q_target << 0, 0, 0.793,
                0, 0, 0, 1,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 1.1,
                0, 0, 0, 1.1;

    mpc.SetConstantConfigTarget(q_target);

    v_target.resize(g1.GetVelDim());
    v_target << 0., 0, 0,
                0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0;
    mpc.SetConstantVelTarget(v_target);

    Trajectory traj;
    traj.UpdateSizes(g1.GetConfigDim(), g1.GetVelDim(), g1.GetNumInputs(), mpc.GetContactFrames(), mpc.GetNumNodes());
    traj.SetDefault(q_target);

    traj.SetDtVector(mpc.GetDtVector());

    std::cout << "robot mass: " << g1.GetMass() << std::endl;
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
                traj.SetForce(i, frame, {0, 0, 9.81*g1.GetMass()/num_contacts});
            }
        }

        time += traj.GetDtVec()[i];
    }

    mpc.SetWarmStartTrajectory(traj);

    // std::cout << "press any key to start mpc... " << std::endl;
    // std::cin.get();

    g1.SecondOrderFK(q_target, v_target);
    for (const auto& frame : mpc.GetContactFrames()) {
        std::cout << "frame: " << frame << "\npos: " << g1.GetFrameState(frame).placement.translation().transpose() << std::endl;
        std::cout << "vel: " << g1.GetFrameState(frame).vel.linear().transpose() << std::endl;
    }

    // mpc.Compute(q_target, v_target, traj);

    mpc.GenerateCostReference(q_target,  v_target, v_target.head<3>(), cs);
    std::cout << "\nTargets:" << std::endl;
    for (int i = 0; i < mpc.GetConfigTargets().GetNumNodes(); i++) {
        std::cout << "i: " << i << ", target: " << mpc.GetConfigTargets()[i].transpose() << std::endl;
    }

    // q_target(0) += 0.2;
    mpc.ComputeNLP(q_target, v_target, traj);

    for (int i = 0; i < 20; i++) {
    // TODO: put back!
        vectorx_t q_current;
        traj.GetConfigInterp(0.01, q_current);
        std::cout << "q: " << q_current.transpose() << std::endl;
        vectorx_t v_current;
        traj.GetVelocityInterp(0.01, v_current);

        // cs.ShiftSwings(-0.01);
        mpc.UpdateContactScheduleAndSwingTraj(cs, apex_height, foot_height, apex_time);

        // mpc.ShiftWarmStart(0.01);

        mpc.GenerateCostReference(q_target,  v_target, v_target.head<3>(), cs);

        mpc.Compute(q_current, v_current, traj);
        // mpc.Compute(q_target, v_target, traj);

        double temp_time = 0;
        bool bad_force = false;
        for (int node = 0; node < traj.GetNumNodes(); node++) {
            for (const auto& frame : traj.GetContactFrames()) {
                if (!mpc.PlannedContact(frame, node)) { //!cs.InContact(frame, temp_time)) {
                    vector3_t force_out;
                    // traj.GetForceInterp(temp_time, frame, force_out);
                    force_out = traj.GetForce(node, frame);
                    if (force_out.norm() > 1) {
                        std::cerr << "Node: " << node << ", time: " << temp_time << std::endl;
                        std::cerr << "Force: " << force_out.transpose() << std::endl;
                        std::cerr << "Force norm: " << force_out.norm() << std::endl;
                        bad_force = true;
                    }
                }
            }
            temp_time += traj.GetDtVec()[node];
        }

        if (bad_force) {
            for (int i = 0; i < traj.GetNumNodes(); i++) {
                std::cout << "Node: " << i << std::endl;
                std::cout << "config: " << traj.GetConfiguration(i).transpose() << std::endl;
                std::cout << "vel: " << traj.GetVelocity(i).transpose() << std::endl;
                // std::cout << "torque: " << traj.GetTau(i).transpose() << std::endl;
                g1.SecondOrderFK(traj.GetConfiguration(i), traj.GetVelocity(i));
                for (const auto& frame : mpc.GetContactFrames()) {
                    std::cout << "frame: " << frame << "\npos: " << g1.GetFrameState(frame).placement.translation().transpose() << std::endl;
                    std::cout << "vel: " << g1.GetFrameState(frame).vel.linear().transpose() << std::endl;
                    std::cout << "force: " << traj.GetForce(i, frame).transpose() << std::endl;
                }
                std::cout << std::endl;
            }

            mpc.PrintContactSchedule();
            // throw std::runtime_error("Force norm too large!");
        }

        std::cout << "q current: " << q_current.transpose() << std::endl;
        std::cout << "v current: " << v_current.transpose() << std::endl;
    }



    // for (int i = 0; i < traj.GetNumNodes(); i++) {
    //     std::cout << "Node: " << i << std::endl;
    //     std::cout << "config: " << traj.GetConfiguration(i).transpose() << std::endl;
    //     std::cout << "vel: " << traj.GetVelocity(i).transpose() << std::endl;
    //     std::cout << "torque: " << traj.GetTau(i).transpose() << std::endl;
    //      g1.SecondOrderFK(traj.GetConfiguration(i), traj.GetVelocity(i));
    //      for (const auto& frame : mpc.GetContactFrames()) {
    //          std::cout << "frame: " << frame << "\npos: " << g1.GetFrameState(frame).placement.translation().transpose() << std::endl;
    //          std::cout << "vel: " << g1.GetFrameState(frame).vel.linear().transpose() << std::endl;
    //          std::cout << "force: " << traj.GetForce(i, frame).transpose() << std::endl;
    //      }
    //     std::cout << std::endl;
    // }

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