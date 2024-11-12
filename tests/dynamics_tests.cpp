//
// Created by zolkin on 9/5/24.
//

#include <mujoco.h>

#include "full_order_mpc.h"
#include "full_order_rigid_body.h"

int main() {
using namespace torc::mpc;
using namespace torc::models;

// Make the robot
	std::filesystem::path achilles_urdf = "/home/zolkin/AmberLab/Project-Sample-Walking/sample-contact-walking/achilles_model_custom/urdf/achilles.urdf";
	torc::models::FullOrderRigidBody achilles("achilles", achilles_urdf);

	std::cout << "Mass: " << achilles.GetMass() << std::endl;
// Make the Mpc
	std::filesystem::path mpc_config = "/home/zolkin/AmberLab/Project-Sample-Walking/sample-contact-walking/sample_contact_walking/configs/mpc_config.yaml";
	FullOrderMpc mpc("dynamics_test_achilles_mpc", mpc_config, achilles_urdf);
	mpc.Configure();

// Make the mujoco robot
	std::filesystem::path mujoco_xml = "/home/zolkin/AmberLab/Project-TORC/torc/tests/test_data/achilles.xml";

	char error[1000] = "Could not load binary model";
	const auto model = mj_loadXML(mujoco_xml.c_str(), 0, error, 1000);
	if (!model) {
		throw std::runtime_error("Could not load model");
	}
	auto data = mj_makeData(model);
	if (!data) {
		throw std::runtime_error("Could not make data");
	}

	if (model->nu != achilles.GetNumInputs()*3) {
		throw std::runtime_error("Wrong number of inputs in mujoco!");
	}
	std::cout << "Number of mujoco bodies: " << model->nbody << std::endl;

// Solve MPC
vectorx_t q_ic(achilles.GetConfigDim());
vectorx_t v_ic(achilles.GetVelDim());

q_ic << 0., 0., 0.93,
		0., 0., 0., 1.,
		0.0000, 0.0209, -0.5515,
		1.0239, -0.4725,
		0.0000, 0.0000, 0.0000,
		0.0000,
		0.0000, -0.0209, -0.3200,
		0.9751, -0.6552,
		0.0000, 0.0000, 0.0000,
		0.0000;

v_ic << 0., 0., 0.,
		0., 0., 0.,
		0., 0., 0.,
		0., 0.,
		0., 0., 0., 0.,
		0., 0., 0.,
		0., 0.,
		0., 0., 0., 0.;

vectorx_t q_target = q_ic;
vectorx_t v_target = v_ic;

mpc.SetConstantConfigTarget(q_target);
mpc.SetConstantVelTarget(v_target);

ContactSchedule cs;
cs.SetFrames(mpc.GetContactFrames());
mpc.UpdateContactScheduleAndSwingTraj(cs, 0.08, 0.015, 0.7);

Trajectory traj;
traj.UpdateSizes(achilles.GetConfigDim(), achilles.GetVelDim(), achilles.GetNumInputs(), mpc.GetContactFrames(), mpc.GetNumNodes());
traj.SetDefault(q_ic);
traj.SetDtVector(mpc.GetDtVector());
mpc.SetWarmStartTrajectory(traj);

mpc.ComputeNLP(q_ic, v_ic, traj);

mpc.PrintStatistics();

std::vector<double> pin_fw_diff;
std::vector<double> mujoco_fw_diff;

std::cout << std::fixed << std::setprecision(6);

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

	if (i < traj.GetNumNodes() - 1) {
		std::cout << "MPC forward dynamics: " << std::endl;
		vectorx_t a_mpc = (traj.GetVelocity(i+1) - traj.GetVelocity(i))/traj.GetDtVec()[i];
		std::cout << "a: " << (a_mpc).transpose() << std::endl;

		std::cout << "Pinocchio forward dynamics: " << std::endl;
		const vectorx_t q = traj.GetConfiguration(i);
		const vectorx_t v = (traj.GetVelocity(i) + traj.GetVelocity(i + 1))/2;
		vectorx_t tau = traj.GetTau(i);
		if (tau.size() != achilles.GetNumInputs()) {
			throw std::logic_error("Torque is of invalid size!");
		}

		std::vector<torc::models::ExternalForce<double>> f_ext;
		for (const auto& frame : mpc.GetContactFrames()) {
		    f_ext.emplace_back(frame, traj.GetForce(i, frame));
		}


		vectorx_t xdot = achilles.GetDynamics(q, v, tau, f_ext);
		std::cout << "a: " << xdot.tail(achilles.GetVelDim()).transpose() << std::endl;

		pin_fw_diff.emplace_back((xdot.tail(achilles.GetVelDim()) - a_mpc).norm());

		// double dt = traj.GetDtVec()[i];
		// vectorx_t q_next = vectorx_t::Zero(q_ic.size());
		// pinocchio::integrate(achilles.GetModel(), q_ic, dt*xdot.head(v_ic.size()), q_next);
		// vectorx_t v_next = v_ic + dt*xdot.tail(v_ic.size());

		// std::cout << "Integrated dynamics: " << std::endl;
		// std::cout << "q next: " << q_next.transpose() << std::endl;
		// std::cout << "v next: " << v_next.transpose() << std::endl << std::endl;

		std::cout << "Mujoco forward dynamics: " << std::endl;
		// Set q and v
		// Change the quaternion to match mujoco convention
		vectorx_t q_mj = q;
		q_mj(3) = q(6);
		q_mj(4) = q(3);
		q_mj(5) = q(4);
		q_mj(6) = q(5);

		mju_copy(data->qpos, q_mj.data(), q.size());
		mju_copy(data->qvel, v.data(), v.size());

		// Set controls
		vectorx_t ctrl_temp(tau.size()*3);
		ctrl_temp << q.tail(achilles.GetNumInputs()), v.tail(achilles.GetNumInputs()), tau;
		// TODO: When the gains are set to 0, the bottom two should lead to identical results
		// mju_copy(data->ctrl, ctrl_temp.data(), ctrl_temp.size());
		mju_copy(data->ctrl + tau.size()*2, tau.data(), tau.size());
		mj_forward(model, data);
		Eigen::Map<vectorx_t> a(data->qacc, achilles.GetVelDim());
		std::cout << "a: " << a.transpose() << std::endl;

		mujoco_fw_diff.emplace_back((a - a_mpc).norm());

		std::array<std::string, 4> names = {"right_toe_force_sensor", "right_heel_force_sensor",
		 	"left_toe_force_sensor", "left_heel_force_sensor"};
		std::map<std::string, std::string> sensors_to_frames = {{"right_toe_force_sensor", "foot_front_right"},
			{"right_heel_force_sensor", "foot_rear_right"}, {"left_toe_force_sensor", "foot_front_left"},
			 {"left_heel_force_sensor", "foot_rear_left"}};
		std::map<std::string, std::string> sensors_to_sites = {{"right_toe_force_sensor", "right_toe_force_site"},
			{"right_heel_force_sensor", "right_heel_force_site"}, {"left_toe_force_sensor", "left_toe_force_site"},
			 {"left_heel_force_sensor", "left_heel_force_site"}};
		// Read force sensors
		std::vector<vector3_t> forces;
		for (const auto& sensor : names) {
			int sensor_idx = mj_name2id(model, mjOBJ_SENSOR, sensor.c_str());
			int sensor_adr = model->sensor_adr[sensor_idx];
			if (sensor_idx == -1) {
				throw std::runtime_error("Sensor not found " + sensor);
			}

			vector3_t force_temp = {data->sensordata[sensor_adr],
				data->sensordata[sensor_adr + 1], data->sensordata[sensor_adr + 2]};

			// Mujoco site location
			int site_id = mj_name2id(model, mjOBJ_SITE, sensors_to_sites[sensor].c_str());
			if (site_id == -1) {
				throw std::runtime_error("Site not found " + sensors_to_sites[sensor]);
			}
			std::cout << "site_id: " << site_id << std::endl;
			vector3_t site_pos;
			mju_copy(site_pos.data(), data->site_xpos + 3*site_id, 3);
			std::cout << "mujoco site pos: " << site_pos.transpose() << std::endl;

			matrix3_t site_R;
			mju_copy(site_R.data(), data->site_xmat + 9*site_id, 9);
			std::cout << "mujoco site R: \n" << site_R << std::endl;

			// Check site relative to pinocchio
			achilles.FirstOrderFK(q);
			vector3_t pin_contact_site = achilles.GetFrameState(sensors_to_frames[sensor]).placement.translation();
			std::cout << "pin contact site: " << pin_contact_site.transpose() << std::endl;

			forces.emplace_back(site_R*force_temp);
		}

		std::cout << "Mujoco contact forces: " << std::endl;
		for (const auto& force : forces) {
			std::cout << force.transpose() << std::endl;
		}



		// mujoco inverse dynamics
		// set everything
		// mju_copy(data->qpos, q.data(), q.size());
		// mju_copy(data->qvel, v.data(), v.size());
		// mju_copy(data->qacc, a_mpc.data(), a_mpc.size());

		// vectorx_t mj_qrfc_before(achilles.GetNumInputs());
		// mju_copy(mj_qrfc_before.data(), data->qfrc_inverse, achilles.GetNumInputs());
		// std::cout << "Net external force before ID: " << mj_qrfc_before.transpose() << std::endl;

		// Set body forces
		// for (const auto& frame : mpc.GetContactFrames()) {
		// 	int body_idx = mj_name2id(model, mjOBJ_BODY, frame.c_str());
		// 	if (body_idx == -1) {
		// 		throw std::runtime_error("Could not find the body: " + frame);
		// 	}
		//
		// 	// TODO: What frame should these forces be expressed in?
		// 	// TODO: verify that the linear forces are first
		// 	// mju_copy(data->xfrc_applied + body_idx*6, traj.GetForce(i, frame).data(), 3);
		// }

		// mj_fwdActuation(model, data);

		// mj_inverse(model, data);
		//
		// vectorx_t actuator_torque_after(achilles.GetVelDim());
		// mju_copy(actuator_torque_after.data(), data->qfrc_actuator, achilles.GetVelDim());
		// std::cout << "Mujoco qfrc_actuator: " << actuator_torque_after.transpose() << std::endl;
		//
		// vectorx_t actuator_torque_nu(achilles.GetNumInputs());
		// mju_copy(actuator_torque_nu.data(), data->actuator_force, achilles.GetNumInputs());
		// std::cout << "Mujoco actuator forces (actuator space): " << actuator_torque_nu.transpose() << std::endl;
		//
		// vectorx_t mj_constraint_force(achilles.GetVelDim());
		// mju_copy(mj_constraint_force.data(), data->qfrc_constraint, achilles.GetVelDim());
		// std::cout << "Mujoco constraint forces: " << mj_constraint_force.transpose() << std::endl;
		//
		// vectorx_t mj_qfrc_net_after(achilles.GetVelDim());
		// mju_copy(mj_qfrc_net_after.data(), data->qfrc_inverse, achilles.GetVelDim());
		// std::cout << "Net external force after ID: " << mj_qfrc_net_after.transpose() << std::endl;
		//
		// vectorx_t mpc_net_force = vectorx_t::Zero(achilles.GetVelDim());
		// mpc_net_force.tail(achilles.GetNumInputs()) = traj.GetTau(i);
		// std::cout << "MPC next external force: " << mpc_net_force.transpose() << std::endl;
		// // TODO: Add in external forces and make the MPC match the mujoco

		std::cout << "MPC integration: " << std::endl;

		std::cout << "Pinocchio integration: " << std::endl;

		mj_step(model, data);
		vectorx_t mj_int_q(achilles.GetConfigDim());
		mju_copy(mj_int_q.data(), data->qpos, achilles.GetConfigDim());
		std::cout << "Mujoco integration config: " << mj_int_q.transpose() << std::endl;
		vectorx_t mj_int_v(achilles.GetVelDim());
		mju_copy(mj_int_v.data(), data->qvel, achilles.GetVelDim());
		std::cout << "Mujoco integration vel: " << mj_int_v.transpose() << std::endl;

		std::cout << std::endl;
	}
}

std::cout << "Pinocchio difference: " << std::endl;
for (const auto& diff : pin_fw_diff) {
	std::cout << diff << " ";
}
std::cout << std::endl;

std::cout << "Mujoco difference: " << std::endl;
for (const auto& diff : mujoco_fw_diff) {
	std::cout << diff << " ";
}

std::cout << std::endl;

mj_deleteData(data);
mj_deleteModel(model);


// Check dynamics and integrated dynamics between MPC, pinocchio forward and Mujoco step
}