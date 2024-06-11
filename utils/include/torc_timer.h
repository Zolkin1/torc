//
// Created by zolkin on 6/11/24.
//

#ifndef TORC_TORC_TIMER_H
#define TORC_TORC_TIMER_H

#include <chrono>

namespace torc::utils {
    class TORCTimer {
    public:
        void Tic();
        void Toc();

        template<class time_type>
        time_type Duration() const;
    protected:
    private:
        std::chrono::steady_clock::time_point start_;
        std::chrono::steady_clock::time_point end_;
    };
} // torc::utils


#endif //TORC_TORC_TIMER_H
