//
// Created by zolkin on 1/28/25.
//

#include "pinocchio_interface.h"
#include "SRBConstraint.h"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/model.hpp>

namespace torc::mpc {
    SRBConstraint::SRBConstraint(int first_node, int last_node, const std::string &name,
        const std::vector<std::string>& contact_frames, const std::filesystem::path &deriv_lib_path, bool compile_derivs,
        const models::FullOrderRigidBody &model, const vectorx_t& q_nom)
        : Constraint(first_node, last_node, name), model_(model) {

        vel_dim_ = model_.GetVelDim();
        config_dim_ = model_.GetConfigDim();
        tau_dim_ = model_.GetVelDim() - FLOATING_VEL;
        num_contacts_ = contact_frames.size();
        contact_frames_ = contact_frames;

        // TODO: Compute intertia and COM offset for the fixed (nominal) joints
        pinocchio::Data pin_data(model_.GetModel());
        vector3_t com_pos = pinocchio::centerOfMass(model_.GetModel(), pin_data);
        base_to_com_ = com_pos - q_nom.head<POS_VARS>();
        base_to_com_skew_sym_ << 0, -base_to_com_[2], base_to_com_[1],
                base_to_com_[2], 0, -base_to_com_[0],
                -base_to_com_[1], base_to_com_[0], 0;

        for (int i = 0; i < 10; i++) {
            std::cout << "joint " << i << " inertia:" << std::endl;
            std::cout << "name: " << model_.GetJointName(i) << std::endl;
            std::cout << model_.GetModel().inertias[i].inertia().matrix() << std::endl;
        }

        // Make a new model with all the joints locked
        std::vector<pinocchio::JointIndex> joint_ids;
        vectorx_t q = pinocchio::neutral((model.GetModel()));
        for (int i = 0; i < model_.GetNumJoints(); i++) {
            if (model_.GetJointName(i) != "universe" && model_.GetJointName(i) != "root_joint") {
                joint_ids.push_back(model_.GetModel().getJointId(model_.GetJointName(i)));
                q[i] = q_nom[i + 5]; // Skip over the floating base joint
            }
        }

        srb_pin_model_ = pinocchio::buildReducedModel(model_.GetModel(), joint_ids, q);
        srb_pin_data_ = std::make_unique<pinocchio::Data>(srb_pin_model_);

        ad_srb_pin_model_ = srb_pin_model_.cast<torc::ad::adcg_t>();
        ad_srb_pin_data_ = std::make_shared<models::ad_pin_data_t>(ad_srb_pin_model_);

        if (ad_srb_pin_model_.njoints != 2) {
            throw std::runtime_error("SRB model does not have exactly two joints!");
        }

        if (ad_srb_pin_model_.nv != FLOATING_VEL) {
            throw std::runtime_error("SRB model does have have 6 dim vel!");
        }


        dynamics_function_ = std::make_unique<ad::CppADInterface>(
            std::bind(&SRBConstraint::SRBDynamics, this, contact_frames, std::placeholders::_1,
                      std::placeholders::_2, std::placeholders::_3),
            name_ + "_dynamics_constraint",
            deriv_lib_path,
            ad::DerivativeOrder::FirstOrder, 3 * vel_dim_ + CONTACT_3DOF * num_contacts_,
            1 + config_dim_ + 2 * vel_dim_ + CONTACT_3DOF * num_contacts_,
            true //compile_derivs
        );

        integration_function_ = std::make_unique<ad::CppADInterface>(
            std::bind(&SRBConstraint::IntegrationConstraint, this, std::placeholders::_1, std::placeholders::_2,
                      std::placeholders::_3),
            name_ + "_integration_constraint",
            deriv_lib_path,
            ad::DerivativeOrder::FirstOrder, 2 * vel_dim_, 1 + 2 * config_dim_ + vel_dim_,
            true //compile_derivs
        );
    }

    void SRBConstraint::SRBDynamics(const std::vector<std::string> &frames,
                                    const ad::ad_vector_t &dqk_dvk_dvkp1_dfk, const ad::ad_vector_t &qk_vk_vkp1_fk_dt,
                                    ad::ad_vector_t &violation) {

        // Decision variables
        const ad::ad_vector_t& dqk = dqk_dvk_dvkp1_dfk.head(vel_dim_);
        const ad::ad_vector_t& dvk = dqk_dvk_dvkp1_dfk.segment(vel_dim_, vel_dim_);
        const ad::ad_vector_t& dvkp1 = dqk_dvk_dvkp1_dfk.segment(2*vel_dim_, vel_dim_);
        const ad::ad_vector_t& dfk = dqk_dvk_dvkp1_dfk.segment(3*vel_dim_, CONTACT_3DOF*num_contacts_);

        // Reference trajectory
        const ad::ad_vector_t& qk = qk_vk_vkp1_fk_dt.head(config_dim_);
        const ad::ad_vector_t& vk = qk_vk_vkp1_fk_dt.segment(config_dim_, vel_dim_);
        const ad::ad_vector_t& vkp1 = qk_vk_vkp1_fk_dt.segment(config_dim_ + vel_dim_, vel_dim_);
        const ad::ad_vector_t& fk = qk_vk_vkp1_fk_dt.segment(config_dim_ + 2*vel_dim_, CONTACT_3DOF*num_contacts_);
        const ad::adcg_t& dt = qk_vk_vkp1_fk_dt(config_dim_ + 2*vel_dim_ + CONTACT_3DOF*num_contacts_);

        // Current values
        const ad::ad_vector_t qk_curr = models::ConvertdqToq(dqk, qk);
        ad::ad_vector_t vk_curr = (dvk + vk).head<FLOATING_VEL>();
        const ad::ad_vector_t vkp1_curr = (dvkp1 + vkp1).head<FLOATING_VEL>();
        const ad::ad_vector_t fk_curr = dfk + fk;

        // Intermediate values
        ad::ad_vector_t a = (vkp1_curr - vk_curr)/dt; // TODO: Do I need to account for the different local frames somehow?
        std::vector<models::ExternalForce<ad::adcg_t>> f_ext;

        int idx = 0;
        for (const auto& frame : frames) {
            f_ext.emplace_back(frame, fk_curr.segment<CONTACT_3DOF>(idx));
            idx += CONTACT_3DOF;
        }

        // Compute the contact locations

        // Compute the forces in the correct frames
        pinocchio::DataTpl<torc::ad::adcg_t> data(model_.GetADPinModel());
        pinocchio::framesForwardKinematics(model_.GetADPinModel(), data, qk_curr);

        pinocchio::framesForwardKinematics(ad_srb_pin_model_, *ad_srb_pin_data_, qk_curr.head<FLOATING_BASE>());

        pinocchio::container::aligned_vector<pinocchio::ForceTpl<ad::adcg_t>> forces(ad_srb_pin_model_.njoints,
            pinocchio::ForceTpl<ad::adcg_t>::Zero());
        for (const auto& f : f_ext) {
            // Get the frame where the contact is
            const long frame_idx = model_.GetADPinModel().getFrameId(f.frame_name);
            // Get the parent frame
            const int jnt_idx = 1; //Use the root joint //model_.GetADPinModel().frames.at(frame_idx).parentJoint;

            // Get the translation from the joint frame to the contact frame
            const ad::ad_vector3_t translationContactToJoint = data.oMf[frame_idx].translation() - qk_curr.head<POS_VARS>();

            // Get the rotation from the world frame to the joint frame
            const ad::ad_matrix3_t rotationWorldToJoint = data.oMi[jnt_idx].rotation().transpose();

            // Get the contact forces in the joint frame
            const ad::ad_vector3_t contact_force = rotationWorldToJoint * f.force_linear;
            forces.at(jnt_idx).linear() += contact_force;

            // Calculate the angular (torque) forces
            forces.at(jnt_idx).angular() += translationContactToJoint.cross(contact_force);
        }

        // Compute error
        ad::ad_vector_t qk_curr_head = qk_curr.head<FLOATING_BASE>();
        const ad::ad_vector_t tau_id = pinocchio::rnea(ad_srb_pin_model_, *ad_srb_pin_data_,
            qk_curr_head, vk_curr, a, forces);

        violation = tau_id;


        // // Returns the discrete change in velocity
        // const ad::ad_vector_t& dqk = dqk_dvk_dvkp1_dfk.head(vel_dim_);
        // const ad::ad_vector_t& dvk = dqk_dvk_dvkp1_dfk.segment(vel_dim_, vel_dim_);
        // const ad::ad_vector_t& dvkp1 = dqk_dvk_dvkp1_dfk.segment(2*vel_dim_, vel_dim_);
        // const ad::ad_vector_t& dF = dqk_dvk_dvkp1_dfk.tail(CONTACT_3DOF*num_contacts_);
        //
        // const ad::ad_vector_t& qk = qk_qkp1_vk_vkp1_fk_dt.head(config_dim_);
        // const ad::ad_vector_t& qkp1 = qk_qkp1_vk_vkp1_fk_dt.segment(config_dim_, config_dim_);
        // const ad::ad_vector_t& vk = qk_qkp1_vk_vkp1_fk_dt.segment(2*config_dim_, vel_dim_);
        // const ad::ad_vector_t& vkp1 = qk_qkp1_vk_vkp1_fk_dt.segment(2*config_dim_ + vel_dim_, vel_dim_);
        // const ad::ad_vector_t& F = qk_qkp1_vk_vkp1_fk_dt.segment(2*config_dim_ + 2*vel_dim_, CONTACT_3DOF*num_contacts_);
        // const ad::adcg_t& dt = qk_qkp1_vk_vkp1_fk_dt[2*config_dim_ + 2*vel_dim_ + CONTACT_3DOF*num_contacts_];
        //
        // const ad::ad_vector_t qk_curr = models::ConvertdqToq(dqk, qk);
        // // const ad::ad_vector_t qkp1_curr = models::ConvertdqToq(dqkp1, qkp1);
        // const ad::ad_vector_t vk_curr = dvk + vk;
        // const ad::ad_vector_t vkp1_curr = dvkp1 + vkp1;
        // const ad::ad_vector_t F_curr = F + dF;
        //
        // // Forward Kinematics
        // pinocchio::forwardKinematics(model_.GetADPinModel(), *model_.GetADPinData(), qk_curr);
        //
        // ad::ad_vector3_t a_com = ad::ad_vector3_t::Zero();
        // // Rotate gravity into the local frame
        // ad::ad_matrix3_t Rk = static_cast<ad::ad_quat>(qk_curr.segment<4>(3)).toRotationMatrix();
        // // ad::ad_matrix3_t Rkp1 = static_cast<ad::ad_quat>(qkp1_curr.segment<4>(3)).toRotationMatrix();
        // ad::ad_vector3_t g = ad::ad_vector3_t::Zero();
        // g(2) = -9.81;
        //
        // a_com = g;
        //
        // ad::ad_vector3_t dIomega_com = ad::ad_vector3_t::Zero();
        //
        // // Loop through contacts
        // for (int i = 0; i < contact_frames.size(); i++) {
        //     a_com += F_curr.segment<POS_VARS>(i*POS_VARS)/model_.GetMass();
        //
        //     // Get contact location
        //     const long frame_idx1 = model_.GetFrameIdx(contact_frames[i]);
        //     pinocchio::updateFramePlacement(model_.GetADPinModel(), *model_.GetADPinData(), frame_idx1);
        //     ad::ad_vector3_t contact_loc = model_.GetADPinData()->oMf.at(frame_idx1).translation();
        //
        //     // Get COM
        //     // TODO: Could consider calculating the COM based on the current joint config
        //     ad::ad_vector3_t com_pos = qk_curr.head<POS_VARS>() + base_to_com_;
        //
        //     dIomega_com += (contact_loc - com_pos).cross(F_curr.segment<POS_VARS>(i*POS_VARS));
        // }
        //
        // // Now extract the base velocities
        // // COM linear vel is the same as the base linear vel
        // // Angular vel requires inverting the inertia matrix
        // // Get inertia in the local frame
        // ad::ad_matrix3_t Ilocal = model_.GetADPinModel().inertias[1].inertia().matrix();  // TODO: Check that this is the right joint
        // // Now adjust to put into the global frame
        // // Get the rotation matrix for the quaternion
        // ad::ad_vector3_t domega_com = (Rk.transpose()*Ilocal*Rk).inverse()*dIomega_com;
        //
        // // Add the angular term to the linear term to account for the translation
        // ad::ad_vector_t a_base = a_com + base_to_com_skew_sym_ * domega_com;
        // ad::ad_vector_t domega_base = domega_com;
        //
        // ad::ad_vector_t dv_world = dt*a_base;
        // ad::ad_vector_t domega_world = dt*domega_base;
        //
        // // Rotate into the correct frame
        // // ad::ad_matrix3_t base_pos_world_skem_sym;
        // // base_pos_world_skem_sym << ad::adcg_t(0), -qk_curr[2], qk_curr[1],
        // //                          qk_curr[2], ad::adcg_t(0), -qk_curr[0],
        // //                          -qk_curr[1], qk_curr[0], ad::adcg_t(0);
        // // ad::ad_vector_t dv_local = Rk*dv_world + base_pos_world_skem_sym*Rk*domega_world;
        // // ad::ad_vector_t domega_local = Rk*domega_world;
        //
        // // TODO: This is currently global velocity
        // // Now integrate
        // ad::ad_vector3_t dvkp1_base = dv_world + vk_curr.head<POS_VARS>() - vkp1.head<POS_VARS>();
        // ad::ad_vector3_t domegakp1_base = domega_world + (vk_curr.segment<POS_VARS>(POS_VARS) - vkp1.segment<POS_VARS>(POS_VARS));
        //
        // violation.resize(6);
        // violation << dvkp1_base, domegakp1_base;
    }

    void SRBConstraint::IntegrationConstraint(const ad::ad_vector_t& dqk_dvk,
         const ad::ad_vector_t& dt_qkbar_qkp1bar_vk, ad::ad_vector_t& dqkp1) {
        // From the reference trajectory
        const ad::adcg_t& dt = dt_qkbar_qkp1bar_vk(0);
        const ad::ad_vector_t& qkbar = dt_qkbar_qkp1bar_vk.segment(1, config_dim_);
        const ad::ad_vector_t& qkp1bar = dt_qkbar_qkp1bar_vk.segment(1 + config_dim_, config_dim_);
        const ad::ad_vector_t& vkbar = dt_qkbar_qkp1bar_vk.segment(1 + 2*config_dim_, vel_dim_);

        // Changes from decision variables
        const ad::ad_vector_t& dqk = dqk_dvk.head(vel_dim_);
        const ad::ad_vector_t& dvk = dqk_dvk.segment(vel_dim_, vel_dim_);

        // Get the current configuration
        const ad::ad_vector_t qk = torc::models::ConvertdqToq(dqk, qkbar);

        // Velocity
        const ad::ad_vector_t vk = vkbar + dvk;

        ad::ad_vector_t v = dt*vk;

        // // Rotate the velocity into the local frame TODO: Keep or remove?
        // ad::ad_matrix3_t R = static_cast<ad::ad_quat>(qk.segment<4>(3)).toRotationMatrix();
        // v.head<POS_VARS>() = R.transpose()*v.head<POS_VARS>();

        const ad::ad_vector_t qkp1_new = pinocchio::integrate(model_.GetADPinModel(), qk, v);

        // Floating base position differences
        dqkp1.resize(vel_dim_);
        dqkp1.head<POS_VARS>() = qkp1_new.head<POS_VARS>() - qkp1bar.head<POS_VARS>(); //qkp1_new.head<POS_VARS>() - qkp1.head<POS_VARS>();

        // Quaternion difference in the tangent space
        Eigen::Quaternion<ad::adcg_t> quat_kp1_bar(qkp1bar.segment<QUAT_VARS>(POS_VARS));
        Eigen::Quaternion<ad::adcg_t> quat_kp1_new(qkp1_new.segment<QUAT_VARS>(POS_VARS));

        // Eigen's inverse has an if statement, so we can't use it in codegen
        quat_kp1_bar = Eigen::Quaternion<torc::ad::adcg_t>(quat_kp1_bar.conjugate().coeffs() / quat_kp1_bar.squaredNorm());   // Assumes norm > 0
        dqkp1.segment<3>(POS_VARS) = pinocchio::quaternion::log3(quat_kp1_bar * quat_kp1_new);

        // Joint differences
        dqkp1.tail(vel_dim_ - FLOATING_VEL) = qkp1_new.tail(config_dim_ - FLOATING_BASE) - qkp1bar.tail(config_dim_ - FLOATING_BASE);
    }

    int SRBConstraint::GetNumConstraints() const {
        return dynamics_function_->GetRangeSize() + integration_function_->GetRangeSize();
    }

    void SRBConstraint::GetLinDynamics(const vectorx_t &q1_lin, const vectorx_t &q2_lin, const vectorx_t &v1_lin,
        const vectorx_t &v2_lin, const vectorx_t &force_lin, double dt, matrixx_t &A, matrixx_t &B, vectorx_t &b) {
        // --------- Dynamics --------- //
        vectorx_t x_zero = vectorx_t::Zero(dynamics_function_->GetDomainSize());
        vectorx_t p(dynamics_function_->GetParameterSize());
        p << q1_lin, v1_lin, v2_lin, force_lin, dt;

        matrixx_t dyn_jac;
        dynamics_function_->GetJacobian(x_zero, p, dyn_jac);

        vectorx_t fbar;
        dynamics_function_->GetFunctionValue(x_zero, p, fbar);

        // --------- Integration --------- //
        x_zero.resize(integration_function_->GetDomainSize());
        x_zero.setZero();

        p.resize(integration_function_->GetParameterSize());
        p << dt, q1_lin, q2_lin, v1_lin;

        matrixx_t int_jac;
        integration_function_->GetJacobian(x_zero, p, int_jac);

        vectorx_t int_fbar;
        integration_function_->GetFunctionValue(x_zero, p, int_fbar);

        A.setZero();
        B.setZero();
        b.setZero();

        // matrixx_t dq2_inv = int_jac.middleCols(vel_dim_, vel_dim_).inverse();
        // std::cout << "dq2_inv:\n" << dq2_inv << std::endl;

        // Integration
        A.topRows(vel_dim_) << int_jac.leftCols(vel_dim_), int_jac.middleCols(vel_dim_, FLOATING_VEL);
        B.topLeftCorner(vel_dim_, vel_dim_ - FLOATING_VEL) = int_jac.rightCols(vel_dim_ - FLOATING_VEL);

        // Dynamics
        matrixx_t dv2_inv = dyn_jac.middleCols(2*vel_dim_, FLOATING_VEL).inverse();

        A.bottomRows<FLOATING_VEL>() << -dv2_inv*dyn_jac.leftCols(vel_dim_ + FLOATING_VEL);
        B.bottomRows<FLOATING_VEL>() << -dv2_inv*dyn_jac.middleCols(vel_dim_ + FLOATING_VEL, vel_dim_ - FLOATING_VEL),
            -dv2_inv*dyn_jac.rightCols(num_contacts_*CONTACT_3DOF);

        b << int_fbar, -dv2_inv*fbar;
    }

    std::pair<vectorx_t, vectorx_t> SRBConstraint::GetViolation(const vectorx_t &q1_lin, const vectorx_t &q2_lin,
        const vectorx_t &v1_lin, const vectorx_t &v2_lin, const vectorx_t &force_lin,
        double dt, const vectorx_t &dq1, const vectorx_t &dq2, const vectorx_t &dv1, const vectorx_t &dv2,
        const vectorx_t &dforce) {

        vectorx_t x(dynamics_function_->GetDomainSize());
        vectorx_t p(dynamics_function_->GetParameterSize());
        vectorx_t dyn_vio(dynamics_function_->GetRangeSize());

        x << dq1, dv1, dv2, dforce;
        p << q1_lin, v1_lin, v2_lin, force_lin, dt;
        dynamics_function_->GetFunctionValue(x, p, dyn_vio);
        // dyn_vio = dyn_vio - dv2.head<FLOATING_VEL>();

        vectorx_t int_violation(integration_function_->GetRangeSize());
        x.resize(integration_function_->GetDomainSize());
        p.resize(integration_function_->GetParameterSize());

        x << dq1, dv1;
        p << dt, q1_lin, q2_lin, v1_lin;
        integration_function_->GetFunctionValue(x, p, int_violation);
        int_violation = int_violation - dq2;

        return {int_violation, dyn_vio};
    }


}
