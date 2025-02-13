//
// Created by zolkin on 2/8/25.
//
// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>
#include <cstring>
#include <pthread.h>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <torc_timer.h>

#include "CentroidalDynamicsConstraint.h"
#include "DynamicsConstraint.h"
#include "ForwardKinematicsCost.h"
#include "hpipm_mpc.h"
#include "reference_generator.h"
#include "SRBConstraint.h"
#include "step_planner.h"

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;
bool user_pause = false;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
  // backspace: reset simulation
    if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
    }

    if (act==GLFW_PRESS && key==GLFW_KEY_SPACE) {
        user_pause = !user_pause;
    }

}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
  // update button state
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

void LogEigenVec(std::ofstream& log_file, const torc::mpc::vectorx_t& x) {
    for (int i = 0; i < x.size(); i++) {
        log_file << x(i) << ",";
    }
}

// main function
int main(int argc, const char** argv) {

    struct sched_param param;
    param.sched_priority = 99;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

    const std::string mujoco_xml = "/home/zolkin/AmberLab/Project-Sample-Walking/sample-contact-walking/g1_model/mujoco/basic_scene.xml";

    // load and compile model
    char error[1000] = "Could not load binary model";
    m = mj_loadXML(mujoco_xml.c_str(), 0, error, 1000);

    if (!m) {
    mju_error("Load model error: %s", error);
    }

    // make data
    d = mj_makeData(m);

    // init GLFW
    if (!glfwInit()) {
    mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // ------------------------------------------------------------ //
    // -------------------- Make the MPC -------------------------- //
    // ------------------------------------------------------------ //

    // Create all the constraints
    using namespace torc::mpc;
    std::filesystem::path g1_urdf = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_hand_v2.urdf";

    std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/g1_mpc_config_2.yaml";

    MpcSettings settings(mpc_config);
    settings.Print();

    std::vector<std::pair<std::string, std::string>> poly_contact_frames;
    poly_contact_frames.emplace_back(settings.polytope_frames[0], settings.contact_frames[0]);
    poly_contact_frames.emplace_back(settings.polytope_frames[0], settings.contact_frames[1]);
    poly_contact_frames.emplace_back(settings.polytope_frames[1], settings.contact_frames[2]);
    poly_contact_frames.emplace_back(settings.polytope_frames[1], settings.contact_frames[3]);

    settings.poly_contact_pairs = poly_contact_frames;

    const std::string pin_model_name = "test_pin_model";

    torc::models::FullOrderRigidBody g1("g1", g1_urdf, settings.joint_skip_names, settings.joint_skip_values);
    torc::models::FullOrderRigidBody g1full("g1full", g1_urdf);

    fs::path deriv_lib_path = fs::current_path();
    deriv_lib_path = deriv_lib_path / "deriv_libs";

    // --------------------------------- //
    // ---------- Constraints ---------- //
    // --------------------------------- //
    // ---------- FO Dynamics ---------- //
    DynamicsConstraint dynamics_constraints(g1, settings.contact_frames, "g1_full_order",
        deriv_lib_path, settings.compile_derivs, 0, settings.nodes_full_dynamics);

    // ---------- Centroidal Dynamics ---------- //
    CentroidalDynamicsConstraint centroidal_dynamics(g1, settings.contact_frames, "g1_centroidal", settings.deriv_lib_path,
        settings.compile_derivs, settings.nodes_full_dynamics, settings.nodes);

    // ---------- SRB Dynamics ---------- //
    SRBConstraint srb_dynamics(settings.nodes_full_dynamics, settings.nodes,
        "g1_srb",
        settings.contact_frames, settings.deriv_lib_path, settings.compile_derivs, g1, settings.q_target);

    // ---------- Box Constraints ---------- //
    // Config
    std::vector<int> config_lims_idxs;
    for (int i = 0; i < g1.GetConfigDim() - torc::mpc::FLOATING_BASE; ++i) {
        config_lims_idxs.push_back(i + FLOATING_VEL);
    }
    BoxConstraint config_box(1, settings.nodes, "config_box",
        g1.GetLowerConfigLimits().tail(g1.GetConfigDim() - torc::mpc::FLOATING_BASE),
        g1.GetUpperConfigLimits().tail(g1.GetConfigDim() - torc::mpc::FLOATING_BASE),
        config_lims_idxs);

    // Vel
    std::vector<int> vel_lims_idxs;
    for (int i = 0; i < g1.GetVelDim() - torc::mpc::FLOATING_VEL; ++i) {
        vel_lims_idxs.push_back(i + FLOATING_VEL);
    }
    std::cout << g1.GetVelocityJointLimits().transpose() << std::endl;
    BoxConstraint vel_box(1, settings.nodes, "vel_box",
        -g1.GetVelocityJointLimits().tail(g1.GetVelDim() - torc::mpc::FLOATING_VEL),
        g1.GetVelocityJointLimits().tail(g1.GetVelDim() - torc::mpc::FLOATING_VEL),
        vel_lims_idxs);

    // Torque
    std::vector<int> tau_lims_idxs;
    for (int i = 0; i < g1.GetVelDim() - torc::mpc::FLOATING_VEL; ++i) {
        tau_lims_idxs.push_back(i);
    }
    BoxConstraint tau_box(0, settings.nodes, "tau_box",
        -g1.GetTorqueJointLimits().tail(g1.GetVelDim() - torc::mpc::FLOATING_VEL),
        g1.GetTorqueJointLimits().tail(g1.GetVelDim() - torc::mpc::FLOATING_VEL),
        tau_lims_idxs);

    // Force
    std::vector<int> force_lim_idxs;
    for (int i = 0; i < 3; i++) {
        force_lim_idxs.push_back(i);
    }
    vectorx_t stance_lb(3), stance_ub(3);
    stance_lb << -1000, -1000, settings.min_grf;
    stance_ub << 1000, 1000, settings.max_grf;
    BoxConstraint stance_force_box(0, settings.nodes, "stance_force_box",
        stance_lb, // Minimum force on the ground
        stance_ub,
        force_lim_idxs);

    vectorx_t swing_lb(3), swing_ub(3);
    swing_lb << 0, 0, 0;
    swing_ub << 0, 0, 0;
    BoxConstraint swing_force_box(0, settings.nodes, "swing_force_box",
        swing_lb, // Minimum force on the ground
        swing_ub,
        force_lim_idxs);

    // ---------- Friction Cone Constraints ---------- //
    FrictionConeConstraint friction_cone_constraint(0,settings.nodes - 1, "friction_cone_cone",
        settings.friction_coef, settings.friction_margin, settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Swing Constraints ---------- //
    SwingConstraint swing_constraint(settings.swing_start_node, settings.swing_end_node,
        "swing_constraint", g1, settings.contact_frames, settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Holonomic Constraints ---------- //
    HolonomicConstraint holonomic_constraint(settings.holonomic_start_node, settings.holonomic_end_node,
        "holonomic_constraint", g1, settings.contact_frames, settings.deriv_lib_path, settings.compile_derivs);

    // ---------- Collision Constraints ---------- //
    CollisionConstraint collision_constraint(settings.collision_start_node, settings.collision_end_node,
        "collision_constraint", g1, settings.deriv_lib_path, settings.compile_derivs, settings.collision_data);

    // ---------- Polytope Constraints ---------- //
    PolytopeConstraint polytope_constraint(settings.polytope_start_node, settings.polytope_end_node, "polytope_constraint",
        settings.polytope_frames, settings.deriv_lib_path, settings.compile_derivs, g1);

    std::cout << "===== Constraints Created =====" << std::endl;

    // --------------------------------- //
    // ------------- Costs ------------- //
    // --------------------------------- //
    // ---------- Velocity Tracking ---------- //
    std::cout << settings.cost_data.at(1).weight.transpose() << std::endl;
    LinearLsCost vel_tracking(0, settings.nodes, "vel_tracking",
        settings.deriv_lib_path, settings.compile_derivs, settings.cost_data.at(1).weight.size());

    // ---------- Tau Tracking ---------- //
    LinearLsCost tau_tracking(0, settings.nodes, "tau_tracking",
        settings.deriv_lib_path, settings.compile_derivs, settings.cost_data.at(2).weight.size());

    // ---------- Force Tracking ---------- //
    LinearLsCost force_tracking(0, settings.nodes, "force_tracking",
        settings.deriv_lib_path, settings.compile_derivs, settings.cost_data.at(3).weight.size());

    // ---------- Config Tracking ---------- //
    ConfigTrackingCost config_tracking(0, settings.nodes, "config_tracking", settings.cost_data.at(0).weight.size(),
        settings.deriv_lib_path, settings.compile_derivs, g1);

    // ---------- Forward Kinematics Tracking ---------- //
    // For now they all need the same weight
    ForwardKinematicsCost fk_cost(0, settings.nodes, "fk_cost", settings.cost_data.at(4).weight.size(),
        settings.deriv_lib_path, settings.compile_derivs, g1, settings.contact_frames);

    // --------------------------------- //
    // ------- Contact Schedule -------- //
    // --------------------------------- //
    torc::mpc::ContactSchedule cs(settings.contact_frames);

    cs.InsertSwing("right_toe", 0.1, 0.5);
    cs.InsertSwing("right_heel", 0.1, 0.5);
    cs.InsertSwing("left_toe", 0.5, 0.9);
    cs.InsertSwing("left_heel", 0.5, 0.9);

    cs.InsertSwing("right_toe", 0.9, 1.2);
    cs.InsertSwing("right_heel", 0.9, 1.2);
    cs.InsertSwing("left_toe", 1.2, 1.6);
    cs.InsertSwing("left_heel", 1.2, 1.6);

    cs.InsertSwing("right_toe", 1.6, 2.0);
    cs.InsertSwing("right_heel", 1.6, 2.0);
    cs.InsertSwing("left_toe", 2.0, 2.4);
    cs.InsertSwing("left_heel", 2.0, 2.4);

    cs.InsertSwing("right_toe", 2.4, 2.8);
    cs.InsertSwing("right_heel", 2.4, 2.8);
    cs.InsertSwing("left_toe", 2.8, 3.2);
    cs.InsertSwing("left_heel", 2.8, 3.2);
    // --------------------------------- //
    // -------------- MPC -------------- //
    // --------------------------------- //
    HpipmMpc mpc(settings, g1);

    std::cout << "===== MPC Created =====" << std::endl;
    mpc.SetDynamicsConstraints(std::move(dynamics_constraints));
    mpc.SetCentroidalDynamicsConstraints(std::move(centroidal_dynamics));
    // mpc.SetSrbConstraint(std::move(srb_dynamics));
    mpc.SetConfigBox(config_box);
    mpc.SetVelBox(vel_box);
    mpc.SetTauBox(tau_box);
    mpc.SetForceBox(stance_force_box, swing_force_box);
    mpc.SetFrictionCone(std::move(friction_cone_constraint));
    mpc.SetSwingConstraint(std::move(swing_constraint));
    mpc.SetHolonomicConstraint(std::move(holonomic_constraint));
    mpc.SetCollisionConstraint(std::move(collision_constraint));
    mpc.SetPolytopeConstraint(std::move(polytope_constraint));
    std::cout << "===== MPC Constraints Added =====" << std::endl;

    mpc.SetVelTrackingCost(std::move(vel_tracking));
    mpc.SetTauTrackingCost(std::move(tau_tracking));
    mpc.SetForceTrackingCost(std::move(force_tracking));
    mpc.SetConfigTrackingCost(std::move(config_tracking));
    mpc.SetFowardKinematicsCost(std::move(fk_cost));

    std::cout << "===== MPC Costs Added =====" << std::endl;

    // Initial linearization
    Trajectory traj;
    SimpleTrajectory q_target(g1.GetConfigDim(), settings.nodes);
    SimpleTrajectory v_target(g1.GetVelDim(), settings.nodes);

    // mpc.UpdateContactSchedule(cs);

    double time = 0;

    const std::array<int, 25> mpc_skip_joints = {5, 11,
                       12, 13, 14,
                       19, 20, 21,
                       22, 23, 24, 25, 26, 27, 28,
                       33, 34, 35,
                       36, 37, 38, 39, 40, 41, 42};

    // Load in the correct key frame
    mj_resetDataKeyframe(m, d, 0);
    std::vector<double> shared_data_tmp;
    for (int i = 0; i < m->nu; i++) {
        shared_data_tmp.push_back(d->ctrl[i]);
    }

    std::ofstream log_file;
    log_file.open("mpc_sync_mujoco.csv");
    bool first_loop = true;

    if (m->nq != 18 + FLOATING_BASE) {
        throw std::runtime_error("Mujoco model not 18dof!");
    }

    // Step Planning
    std::vector<torc::mpc::ContactInfo> contact_polytopes;
    vector4_t b;
    b << 10, 10, -10, -10;
    contact_polytopes.emplace_back(matrix2_t::Identity(), b);
    std::vector<double> contact_offsets;
    for (int i = 0; i < settings.num_contact_locations*2; i++) {
        contact_offsets.push_back(0);
    }
    torc::step_planning::StepPlanner step_planner(contact_polytopes, settings.contact_frames, contact_offsets,
        0.1, settings.polytope_delta);

    // Fake time delay
    // COMPUTE_TIME  should be < MPC_PERIOD
    double COMPUTE_TIME = 0.008; //0.006; //0.008;    // Amount of time to compute the MPC from the time the state is grabbed
    double MPC_PERIOD = 0.01; //0.01;       // How often the MPC is asked to re-compute

    std::ofstream timing_log;
    timing_log.open("mpc_timing.csv");

    vectorx_t q_delay, v_delay;     // Record the time-delayed initial conditions here

    while (!glfwWindowShouldClose(window) && d->time < 4.) {
        if (!user_pause) {
            // State when the MPC starts computing
            bool recorded_state = false;

            if (first_loop) {
                // Get the state from Mujoco
                Eigen::Map<vectorx_t> q_map(d->qpos, m->nq);
                Eigen::Map<vectorx_t> v_map(d->qvel, m->nv);

                q_delay = q_map;
                v_delay = v_map;
            }

            vectorx_t q_temp;
            vectorx_t v_temp;
            if (COMPUTE_TIME == 0) {
                // Get the state from Mujoco
                Eigen::Map<vectorx_t> q_map(d->qpos, m->nq);
                Eigen::Map<vectorx_t> v_map(d->qvel, m->nv);

                q_temp = q_map;
                v_temp = v_map;
            } else {
                q_temp = q_delay;
                v_temp = v_delay;
            }

            // Re-order quat
            double w = q_temp(3);
            q_temp(3) = q_temp(4);
            q_temp(4) = q_temp(5);
            q_temp(5) = q_temp(6);
            q_temp(6) = w;


            // Change the velocity frame
            vector3_t v_world_linear = v_temp.head<3>();
            vector3_t v_local_angular = v_temp.segment<3>(3);

            vectorx_t local_vel(6);

            Eigen::Quaterniond base_quat(q_temp(6), q_temp(3), q_temp(4), q_temp(5));
            // TODO: Double check this conversion
            local_vel.head<3>() = base_quat.toRotationMatrix().transpose() * v_world_linear;
            local_vel.tail<3>() = v_local_angular;

            vectorx_t q_mpc(g1.GetConfigDim());
            vectorx_t v_mpc(g1.GetVelDim());

            q_mpc.head<FLOATING_BASE>() = q_temp.head<FLOATING_BASE>();
            v_mpc.head<FLOATING_VEL>() = local_vel;

            // TODO: Only use this when the non-18dof robot is in use
            // // Reduce states down to just the MPC states
            // for (size_t i = 0; i < m->nq - FLOATING_BASE; i++) {
            //     const auto joint_idx = g1.GetJointID(mj_id2name(m, mjOBJ_JOINT, i + 1));
            //     if (joint_idx.has_value()) {
            //         if (joint_idx.value() < 2) {
            //             std::cerr << "Invalid joint name!" << std::endl;
            //         }
            //         q_mpc(joint_idx.value() - 2 + FLOATING_BASE) = q_map(i + FLOATING_BASE);     // Offset for the root and base joints
            //     } else if (!g1full.GetJointID(mj_id2name(m, mjOBJ_JOINT, i + 1)).has_value()) {
            //         std::cerr << "Joint " << mj_id2name(m, mjOBJ_JOINT, i) << " not found in the full robot model!";
            //         throw std::runtime_error("[UpdateXHat] Joint name not found!");
            //     }
            // }
            //
            // for (size_t i = 0; i < m->nv - FLOATING_VEL; i++) {
            //     const auto joint_idx = g1.GetJointID(mj_id2name(m, mjOBJ_JOINT, i + 1));
            //     if (joint_idx.has_value()) {
            //         if (joint_idx.value() < 2) {
            //             std::cerr << "Invalid joint name!" << std::endl;
            //         }
            //         v_mpc(joint_idx.value() - 2 + FLOATING_VEL) = v_map(i +FLOATING_VEL);     // Offset for the root and base joints
            //     } else if (!g1full.GetJointID(mj_id2name(m, mjOBJ_JOINT, i + 1)).has_value()) {
            //         std::cerr << "Joint " << mj_id2name(m, mjOBJ_JOINT, i) << " not found in the full robot model!";
            //         throw std::runtime_error("[UpdateXHat] Joint name not found!");
            //     }
            // }

            q_mpc.tail(m->nq - FLOATING_BASE) = q_temp.tail(m->nq - FLOATING_BASE);
            v_mpc.tail(m->nv - FLOATING_VEL) = v_temp.tail(m->nv - FLOATING_VEL);

            // std::cout << "q: " << q_mpc.transpose() << std::endl;
            // std::cout << "v: " << v_mpc.transpose() << std::endl;

            if (q_mpc.size() != g1.GetConfigDim()) {
                throw std::runtime_error("q_mpc size is wrong!");
            }

            if (v_mpc.size() != g1.GetVelDim()) {
                throw std::runtime_error("v_mpc size is wrong!");
            }

            if (first_loop) {
                q_target.SetAllData(settings.q_target);
                v_target.SetAllData(settings.v_target);
                mpc.SetLinTrajConfig(q_target);
                mpc.SetLinTrajVel(v_target);

                mpc.SetConfigTarget(q_target);
                mpc.SetVelTarget(v_target);

                first_loop = false;
                mpc.CreateQPData();
                mpc.Compute(d->time, q_mpc, v_mpc, traj);

            }

            // Update the contact schedule
            cs.ShiftSwings(-MPC_PERIOD);
            std::map<std::string, std::vector<vector2_t>> nom_footholds, projected_footholds;
            step_planner.PlanStepsHeuristic(q_target, settings.dt, cs,
                nom_footholds, projected_footholds);
            mpc.UpdateContactSchedule(cs);

            mpc.LogMPCCompute(d->time, q_mpc, v_mpc);   // TODO: Should probably add the logging back into the time recording
            torc::utils::TORCTimer prep_timer;
            prep_timer.Tic();
            mpc.CreateQPData();
            prep_timer.Toc();

            torc::utils::TORCTimer feedback_timer;
            feedback_timer.Tic();
            mpc.Compute(d->time, q_mpc, v_mpc, traj);
            feedback_timer.Toc();

            timing_log << d->time << "," << prep_timer.Duration<std::chrono::microseconds>().count()/1000.0 <<
                "," << feedback_timer.Duration<std::chrono::microseconds>().count()/1000.0 << std::endl;

            // TODO: Why does setting this to 0 (rather than COMPUTE_TIME) make the performance better???
            double traj_start_time = 0;// COMPUTE_TIME;  // Compensate for the compute time

            mjtNum simstart = d->time;
            while (d->time - simstart < MPC_PERIOD) {
                // Get control by interpolation
                vectorx_t qctrl, vctrl, tauctrl, F;
                traj.GetConfigInterp(traj_start_time + d->time - simstart, qctrl);
                traj.GetVelocityInterp(traj_start_time + d->time - simstart, vctrl);
                traj.GetTorqueInterp(traj_start_time + d->time - simstart, tauctrl);

                F.resize(12);
                for (int i = 0; i < settings.num_contact_locations; i++) {
                    vector3_t f_temp;
                    traj.GetForceInterp(traj_start_time + d->time - simstart, settings.contact_frames[i], f_temp);
                    F.segment<3>(3*i) = f_temp;
                }

                log_file << d->time << ",";
                LogEigenVec(log_file, qctrl);
                LogEigenVec(log_file, vctrl);
                LogEigenVec(log_file, tauctrl);
                LogEigenVec(log_file, F);
                log_file << std::endl;

                // TODO: Only use when the non-18dof robot is in
                // // Expand states up to the full mujoco model
                // const auto& joint_skip_values = settings.joint_skip_values;
                //
                // std::vector<double> q_vec(qctrl.data(), qctrl.data() + qctrl.size());
                // std::vector<double> v_vec(vctrl.data(), vctrl.data() + vctrl.size());
                // std::vector<double> tau_vec(tauctrl.data(), tauctrl.data() + tauctrl.size());
                //
                // for (int i = 0; i < mpc_skip_joints.size(); i++) {
                //     q_vec.insert(q_vec.begin() + FLOATING_BASE + mpc_skip_joints[i], joint_skip_values[i]);
                //     v_vec.insert(v_vec.begin() + FLOATING_VEL + mpc_skip_joints[i], 0);
                //     tau_vec.insert(tau_vec.begin() + mpc_skip_joints[i], 0);
                // }
                //
                // Eigen::Map<Eigen::VectorXd> q_ctrl_map(q_vec.data(), q_vec.size());
                // Eigen::Map<Eigen::VectorXd> v_ctrl_map(v_vec.data(), v_vec.size());
                // Eigen::Map<Eigen::VectorXd> tau_ctrl_map(tau_vec.data(), tau_vec.size());
                //
                // vectorx_t q = q_ctrl_map;
                // vectorx_t v = v_ctrl_map;
                // vectorx_t tau = tau_ctrl_map;

                vectorx_t q = qctrl;
                vectorx_t v = vctrl;
                vectorx_t tau = tauctrl;

                if (q.size() != m->nq) {
                    throw std::runtime_error("Invalid q size!");
                }

                if (v.size() != m->nv) {
                    throw std::runtime_error("Invalid v size!");
                }

                if (tau.size() != m->nu/3) {
                    throw std::runtime_error("Invalid tau size!");
                }

                int ntau = m->nq - FLOATING_BASE;
                vectorx_t u(3*ntau);

                u << q.tail(ntau), v.tail(ntau), tau.tail(ntau);

                if (u.size() != m->nu) {
                    throw std::runtime_error("u is the wrong size!");
                }

                for (int i = 0; i < m->nu; i++) {
                    d->ctrl[i] = u(i);
                }

                mj_step(m, d);

                // State used my the next MPC is always MPC-COMPUTE after we start using the newest traj
                if (!recorded_state && d->time - simstart >= MPC_PERIOD - COMPUTE_TIME) {
                    recorded_state = true;
                    Eigen::Map<vectorx_t> qcurrent(d->qpos, m->nq);
                    Eigen::Map<vectorx_t> vcurrent(d->qvel, m->nv);

                    q_delay = qcurrent;
                    v_delay = vcurrent;
                }
            }

            // Only render every other solve
            if (mpc.GetSolveCounter() % 1 == 0) {   // TODO: May want to remove this
                // get framebuffer viewport
                mjrRect viewport = {0, 0, 0, 0};
                glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

                // update scene and render
                mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
                mjr_render(viewport, &scn, &con);

                // swap OpenGL buffers (blocking call due to v-sync)
                glfwSwapBuffers(window);

                // process pending GUI events, call GLFW callbacks
                glfwPollEvents();
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));

            // get framebuffer viewport
            mjrRect viewport = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

            // update scene and render
            mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);

            // process pending GUI events, call GLFW callbacks
            glfwPollEvents();
        }
    }

    std::cout << "Done with while loop." << std::endl;

    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);

    // terminate GLFW (crashes with Linux NVidia drivers)
#if defined(__APPLE__) || defined(_WIN32)
  glfwTerminate();
#endif

  return 1;
}