//
// Created by zolkin on 2/15/25.
//

#ifndef LOW_PASS_FITER_H
#define LOW_PASS_FITER_H

#include <vector>
#include <Eigen/Dense>

namespace torc::state_est {
    using vectorx_t = Eigen::VectorXd;

    class LowPassFilter {
    public:
        // Coefs effect the most recent value with the last coefficient and oldest value with the first coefficient
        LowPassFilter(const std::vector<double>& coefs);

        vectorx_t Filter(const vectorx_t& val);

    protected:
        std::vector<double> coefs_;
        std::vector<vectorx_t> past_vals_;
    private:
    };
}


#endif //LOW_PASS_FITER_H
