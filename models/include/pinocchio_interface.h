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

} // namespace torc::models

#endif //PINOCCHIOINTERFACE_H
