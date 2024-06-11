//
// Created by zolkin on 6/11/24.
//

#include "torc_timer.h"

namespace torc::utils {
    void TORCTimer::Tic() {
        start_ = std::chrono::steady_clock::now();
        end_ = start_;
    }

    void TORCTimer::Toc() {
        end_ = std::chrono::steady_clock::now();
    }

    template<class time_type>
    time_type TORCTimer::Duration() const {
        return std::chrono::duration_cast<time_type>(end_ - start_);
    }

    template std::chrono::nanoseconds TORCTimer::Duration<std::chrono::nanoseconds>() const;
    template std::chrono::microseconds TORCTimer::Duration<std::chrono::microseconds>() const;
    template std::chrono::milliseconds TORCTimer::Duration<std::chrono::milliseconds>() const;
    template std::chrono::seconds TORCTimer::Duration<std::chrono::seconds>() const;

} // torc::utils