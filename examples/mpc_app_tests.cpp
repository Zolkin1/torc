//
// Created by zolkin on 7/24/24.
//

#include "full_order_mpc.h"

// TODO: Fix the new constraint violation issues

int main() {
    using namespace torc::mpc;
    std::filesystem::path achilles_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/achilles.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/achilles_mpc_config_sim.yaml";

    FullOrderMpc mpc("achilles_mpc", mpc_config, achilles_urdf);

    mpc.Configure();

    torc::models::FullOrderRigidBody achilles("achilles", achilles_urdf);

    if (achilles.GetJointID("base_to_torso_joint").has_value()) {
        std::cout << "Joint: base_to_torso_joint, idx: " << achilles.GetJointID("base_to_torso_joint").value() << std::endl;
    } else {
        std::cout << "Joint: base_to_torso_joint not found!" << std::endl;
    }

    if (achilles.GetJointID("left_hip_yaw_joint").has_value()) {
        std::cout << "Joint: left_hip_yaw_joint, idx: " << achilles.GetJointID("left_hip_yaw_joint").value() << std::endl;
    }

    torc::mpc::ContactSchedule cs(mpc.GetContactFrames());
    // cs.InsertContact("right_hand", 0, 1);
    // cs.InsertContact("left_hand", 0, 1);

    // cs.InsertContact("right_foot", 0, 1);
    // cs.InsertContact("foot_front_right", 0, 1);
    // cs.InsertContact("foot_rear_right", 0, 1);

    cs.InsertSwing("foot_front_right", 0.3, 0.6);
    cs.InsertSwing("foot_rear_right", 0.3, 0.6);

    const double apex_height = 0.08;
    const std::vector<double> foot_height = {0.015, 0.015, 0.015, 0.015};
    const double apex_time = 0.75;
    mpc.UpdateContactScheduleAndSwingTraj(cs, apex_height, foot_height, apex_time);
    for (const auto& frame : mpc.GetContactFrames()) {
        mpc.PrintSwingTraj(frame);
    }

    // Print swing trajectory to csv for checking
    std::ofstream traj_file;
    traj_file.open("swing_traj.txt");
    double t = 0.2;
    for (int i = 0; i < 100; i++) {
        double height = cs.GetSwingHeight(apex_height, foot_height[0], foot_height[0], apex_time, t, 0.2, 0.5);
        t += 0.3/100;
        traj_file << height << ", " << t << std::endl;
    }
    std::cout << "Apex time: " << apex_time << " foot height: " << foot_height[0] << " start time: " << 0 << " end time: " << 0.3 << std::endl;
    traj_file.close();

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

    achilles.SecondOrderFK(q_target, v_target);
    for (const auto& frame : mpc.GetContactFrames()) {
        std::cout << "frame: " << frame << "\npos: " << achilles.GetFrameState(frame).placement.translation().transpose() << std::endl;
        std::cout << "vel: " << achilles.GetFrameState(frame).vel.linear().transpose() << std::endl;
    }

    // mpc.Compute(q_target, v_target, traj);

    // mpc.GenerateCostReference(q_target,  v_target, v_target.head<3>());
    // std::cout << "\nTargets:" << std::endl;
    // for (int i = 0; i < mpc.GetConfigTargets().GetNumNodes(); i++) {
    //     std::cout << "i: " << i << ", target: " << mpc.GetConfigTargets()[i].transpose() << std::endl;
    // }

    // q_target(0) += 0.2;
    mpc.ComputeNLP(q_target, v_target, traj);

    // for (int node = 0; node < mpc.GetNumNodes(); node++) {
    //     vectorx_t q_ic = traj.GetConfiguration(node);
    //     vectorx_t v_ic = (traj.GetVelocity(node) + traj.GetVelocity(node + 1))/2;
    //     vectorx_t tau = traj.GetTau(node);
    //
    //     std::vector<torc::models::ExternalForce<double>> f_ext;
    //     for (const auto& frame : mpc.GetContactFrames()) {
    //         f_ext.emplace_back(frame, traj.GetForce(0, frame));
    //     }
    //
    //
    //     vectorx_t xdot = achilles.GetDynamics(q_ic, v_ic, tau, f_ext);
    //     double dt = traj.GetDtVec()[node];
    //     vectorx_t q_next = vectorx_t::Zero(q_ic.size());
    //     pinocchio::integrate(achilles.GetModel(), q_ic, dt*xdot.head(v_ic.size()), q_next);
    //     vectorx_t v_next = v_ic + dt*xdot.tail(v_ic.size());
    //
    //     std::cout << "Node: " << node << std::endl;
    //     std::cout << "Integrated dynamics: " << std::endl;
    //     std::cout << "q next: " << q_next.transpose() << std::endl;
    //     std::cout << "v next: " << v_next.transpose() << std::endl << std::endl;
    //
    //     std::cout << "Next node: " << std::endl;
    //     std::cout << "q[1]: " << traj.GetConfiguration(1).transpose() << std::endl;
    //     std::cout << "v[1]: " << traj.GetVelocity(1).transpose() << std::endl << std::endl << std::endl;
    //     // std::cout << "v[1]: " << ((traj.GetVelocity(1) + traj.GetVelocity(2))/2).transpose() << std::endl;
    // }

    for (int i = 0; i < 20; i++) {
    // TODO: put back!
        vectorx_t q_current;
        traj.GetConfigInterp(0.01, q_current);
        std::cout << "q: " << q_current.transpose() << std::endl;
        vectorx_t v_current;
        traj.GetVelocityInterp(0.01, v_current);

        // cs.ShiftSwings(-0.01);
        mpc.UpdateContactScheduleAndSwingTraj(cs, apex_height, foot_height, apex_time);

        mpc.ShiftWarmStart(0.01);

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
                achilles.SecondOrderFK(traj.GetConfiguration(i), traj.GetVelocity(i));
                for (const auto& frame : mpc.GetContactFrames()) {
                    std::cout << "frame: " << frame << "\npos: " << achilles.GetFrameState(frame).placement.translation().transpose() << std::endl;
                    std::cout << "vel: " << achilles.GetFrameState(frame).vel.linear().transpose() << std::endl;
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



    for (int i = 0; i < traj.GetNumNodes(); i++) {
        std::cout << "Node: " << i << std::endl;
        std::cout << "config: " << traj.GetConfiguration(i).transpose() << std::endl;
        std::cout << "vel: " << traj.GetVelocity(i).transpose() << std::endl;
        std::cout << "torque: " << traj.GetTau(i).transpose() << std::endl;
         achilles.SecondOrderFK(traj.GetConfiguration(i), traj.GetVelocity(i));
         for (const auto& frame : mpc.GetContactFrames()) {
             std::cout << "frame: " << frame << "\npos: " << achilles.GetFrameState(frame).placement.translation().transpose() << std::endl;
             std::cout << "vel: " << achilles.GetFrameState(frame).vel.linear().transpose() << std::endl;
             std::cout << "force: " << traj.GetForce(i, frame).transpose() << std::endl;
         }
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