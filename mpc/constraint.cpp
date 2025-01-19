//
// Created by zolkin on 1/18/25.
//

#include "constraint.h"

namespace torc::mpc {
    Constraint::Constraint(int first_node, int last_node) {
        first_node_ = first_node;
        last_node_ = last_node;
    }

    int Constraint::GetFirstNode() const {
        return first_node_;
    }

    int Constraint::GetLastNode() const {
        return last_node_;
    }

}