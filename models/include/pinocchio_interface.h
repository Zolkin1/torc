//
// Created by zolkin on 8/28/24.
//

#ifndef PINOCCHIOINTERFACE_H
#define PINOCCHIOINTERFACE_H

#include <Eigen/Core>
#include <pinocchio/math/quaternion.hpp>
#include <pinocchio/spatial/explog-quaternion.hpp>

namespace torc::models {

    template<typename ScalarT>
    Eigen::Vector<ScalarT, Eigen::Dynamic> ConvertdqToq(const Eigen::Vector<ScalarT, Eigen::Dynamic>& dq,
        const Eigen::Vector<ScalarT, Eigen::Dynamic>& q) {

        Eigen::Vector<ScalarT, Eigen::Dynamic> q_out(q.size());

        // Position variables
        q_out.template head<3>() = q.template head<3>() + dq.template head<3>();

        // Quaternion
        Eigen::Quaternion<ScalarT> quat(q.template segment<4>(3));
        q_out.template segment<4>(3) = (quat * pinocchio::quaternion::exp3(dq.template segment<3>(3))).coeffs();

        // Joints
        q_out.tail(q.size() - 7) = q.tail(q.size() - 7) + dq.tail(q.size() - 7);

        return q_out;
    }

    template<typename ScalarT>
    Eigen::Vector<ScalarT, Eigen::Dynamic> InverseDynamics(const pinocchio::ModelTpl<ScalarT>& pin_model, pinocchio::DataTpl<ScalarT>& data,
        const Eigen::Vector<ScalarT, Eigen::Dynamic>& q,
        const Eigen::Vector<ScalarT, Eigen::Dynamic>& v,
        const Eigen::Vector<ScalarT, Eigen::Dynamic>& a,
        const std::vector<ExternalForce<ad::adcg_t>>& f_ext) {

        const auto forces = ConvertExternalForcesToPin(pin_model, data, q, f_ext);

        return pinocchio::rnea(pin_model, data, q, v, a, forces);
    }

    template<typename ScalarT>
    pinocchio::container::aligned_vector<pinocchio::ForceTpl<ScalarT>> ConvertExternalForcesToPin(
        const pinocchio::ModelTpl<ScalarT>& pin_model, pinocchio::DataTpl<ScalarT>& pin_data,
        const Eigen::Vector<ScalarT, Eigen::Dynamic>& q,
        const std::vector<ExternalForce<ScalarT>>& f_ext) {

        pinocchio::framesForwardKinematics(pin_model, pin_data, q);

        // Convert force to a pinocchio force
        pinocchio::container::aligned_vector<pinocchio::ForceTpl<ScalarT>> forces(pin_model.njoints, pinocchio::ForceTpl<ScalarT>::Zero());
        // TODO: Is this just data.oMi.act(data.f)? Like here: https://github.com/stack-of-tasks/pinocchio/blob/c989669e255715e2fa2504b3226664bf20de6fb5/unittest/rnea-derivatives.cpp#L143
        for (const auto& f : f_ext) {
            // *** Note *** for now I only support 3DOF contacts. To support 6DOF contacts just need to add the additional torques in from the contact (similar to how the linear forces are translated)
            // Get the frame where the contact is
            const long frame_idx = GetFrameIdx<ScalarT>(pin_model, f.frame_name);
            // Get the parent frame
            const int jnt_idx = pin_model.frames.at(frame_idx).parentJoint;

            // Get the translation from the joint frame to the contact frame
            const Eigen::Vector3<ScalarT> translationContactToJoint = pin_model.frames.at(frame_idx).placement.translation();

            // Get the rotation from the world frame to the joint frame
            const Eigen::Matrix3<ScalarT> rotationWorldToJoint = pin_data.oMi[jnt_idx].rotation().transpose();

            // Get the contact forces in the joint frame
            const Eigen::Vector3<ScalarT> contact_force = rotationWorldToJoint * f.force_linear;
            forces.at(jnt_idx).linear() += contact_force;

            // Calculate the angular (torque) forces
            forces.at(jnt_idx).angular() += translationContactToJoint.cross(contact_force);
        }

        return forces;
    }

    template<typename ScalarT>
    long GetFrameIdx(const pinocchio::ModelTpl<ScalarT>& pin_model, const std::string& frame) {
        long idx = pin_model.getFrameId(frame);
        if (idx == pin_model.frames.size()) {
            return -1;
        } else {
            return idx;
        }
    }

} // namespace torc::models

#endif //PINOCCHIOINTERFACE_H
