//
// Created by zolkin on 1/20/25.
//

#include "Cost.h"

namespace torc::mpc {
    Cost::Cost(int first_node, int last_node, const std::string &name)
        : first_node_(first_node), last_node_(last_node), name_(name) {}

    int Cost::GetFirstNode() const {
        return first_node_;
    }

    int Cost::GetLastNode() const {
        return last_node_;
    }

    std::string Cost::GetName() const {
        return name_;
    }

    bool Cost::IsInNodeRange(int node) const {
        return node >= first_node_ && node < last_node_ + 1;    // The + 1 accounts for the boundary node
    }


}