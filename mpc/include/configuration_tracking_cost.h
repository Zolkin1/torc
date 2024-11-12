//
// Created by zolkin on 8/21/24.
//

#ifndef TORC_CONFIGURATION_TRACKING_COST_H
#define TORC_CONFIGURATION_TRACKING_COST_H

#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include "cpp_ad_interface.h"

namespace torc::mpc {
    using vectorx_t = Eigen::VectorXd;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3x_t = Eigen::Matrix3Xd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix43_t = Eigen::Matrix<double, 4, 3>;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    using sp_matrixx_t = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;


    struct ConfigurationTrackingCost {
        void SetWeights();

        void GetCost(const vectorx_t& dq, const vectorx_t& q_reference, const vectorx_t& q_target);

        void GetJacobian(const vectorx_t& dq, const vectorx_t& q_reference, const vectorx_t& q_target, matrixx_t& jac);

        void GetGaussNewton(const vectorx_t& dq, const vectorx_t& q_reference, const vectorx_t& q_target, matrixx_t& hess);

        void GetJacSparsityPattern(matrixx_t& jac);

        void GetHessSparsityPattern(matrixx_t& hess);

    };
} // namespace torc::mpc


#endif //TORC_CONFIGURATION_TRACKING_COST_H
