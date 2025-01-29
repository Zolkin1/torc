//
// Created by zolkin on 1/18/25.
//

#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include "cpp_ad_interface.h"

namespace torc::mpc {
    // enum StateTypes {
    //     CONFIG,
    //     VEL,
    //     FLOATING_VEL
    // };

    using vectorx_t = Eigen::VectorXd;
    using vector2_t = Eigen::Vector2d;
    using vector3_t = Eigen::Vector3d;
    using quat_t = Eigen::Quaterniond;
    using matrixx_t = Eigen::MatrixXd;
    using matrix3x_t = Eigen::Matrix3Xd;
    using matrix3_t = Eigen::Matrix3d;
    using matrix43_t = Eigen::Matrix<double, 4, 3>;
    using matrix6x_t = Eigen::Matrix<double, 6, Eigen::Dynamic>;
    using sp_matrixx_t = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;

    class Constraint {
    public:
        Constraint(int first_node, int last_node, const std::string& name);

        int GetFirstNode() const;
        int GetLastNode() const;

        virtual bool IsInNodeRange(int node) const;

        virtual int GetNumConstraints() const = 0;
    protected:
        int first_node_;
        int last_node_;
        std::string name_;
    private:
    };
}



#endif //CONSTRAINT_H
