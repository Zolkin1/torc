//
// Created by zolkin on 1/18/25.
//

#include "BoxConstraint.h"

namespace torc::mpc {
    BoxConstraint::BoxConstraint(int first_node, int last_node, const std::string& name, const vectorx_t& lb, const vectorx_t& ub,
            const std::vector<int>& idxs)
        : Constraint(first_node, last_node, name), lb_(lb), ub_(ub), idxs_(idxs) {
        if (idxs.size() != lb_.size() || idxs_.size() != ub_.size()) {
            throw std::runtime_error("[Box constraint " + name + "] has a size mis-match.");
        }
    }

    int BoxConstraint::GetNumConstraints() const {
        return idxs_.size();
    }


    void BoxConstraint::SetLowerBound(const vectorx_t &lb) {
        lb_ = lb;
    }

    void BoxConstraint::SetUpperBound(const vectorx_t &ub) {
        ub_ = ub;
    }

    void BoxConstraint::SetIdxs(const std::vector<int> &idxs) {
        idxs_ = idxs;
    }

    vectorx_t BoxConstraint::GetLowerBound(const vectorx_t& x_lin) const {
        return lb_ - x_lin;
    }

    vectorx_t BoxConstraint::GetUpperBound(const vectorx_t& x_lin) const {
        return ub_ - x_lin;
    }

    const std::vector<int>& BoxConstraint::GetIdxs() const {
        return idxs_;
    }



}