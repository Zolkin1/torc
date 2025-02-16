//
// Created by zolkin on 2/15/25.
//

#include "low_pass_filter.h"


namespace torc::state_est {
    LowPassFilter::LowPassFilter(const std::vector<double>& coefs) : coefs_(coefs) {}

    vectorx_t LowPassFilter::Filter(const vectorx_t& val) {
        past_vals_.emplace_back(val);
        while (past_vals_.size() > coefs_.size()) {
            past_vals_.erase(past_vals_.begin());
        }

        // Perform filter
        vectorx_t res = vectorx_t::Zero(val.size());
       for (int i = 0; i < past_vals_.size(); i++) {
            res += coefs_[i] * past_vals_[i];
        }

        return res;
    }
}