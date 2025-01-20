//
// Created by zolkin on 1/18/25.
//

#include "constraint.h"

namespace torc::mpc {
    Constraint::Constraint(int first_node, int last_node, const std::string& name)
        : first_node_(first_node), last_node_(last_node), name_(name) {
    }

    int Constraint::GetFirstNode() const {
        return first_node_;
    }

    int Constraint::GetLastNode() const {
        return last_node_;
    }

    bool Constraint::IsInNodeRange(int node) const {
        return node >= first_node_ && node < last_node_ + 1;    // The + 1 accounts for the boundary node
    }


}