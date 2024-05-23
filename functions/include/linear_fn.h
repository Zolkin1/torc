#ifndef TORC_LINEAR_FN_H
#define TORC_LINEAR_FN_H

#include "explicit_fn.h"

namespace torc::fn {
    /**
     * Class implementation of a linear fn function, f(x) = q^T x
     * @tparam scalar_t the type of scalar used for the fn
     */
    template <class scalar_t>
    class LinearFn: public ExplicitFn<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;

    public:
        /**
         * Overloaded constructor for the LinearFn class.
         * @param coefficients the linear coefficients
         * @param identifier string identifier
         */
        explicit LinearFn(const vectorx_t &coefficients, const std::string &identifier="LinearFnInstance") {
            q_ = coefficients;
            this->SetName(identifier);
            this->dim_ = coefficients.size();

            this->func_ = [this](const vectorx_t& x) { return this->q_.dot(x); };
            this->grad_ = [this](const vectorx_t& x) { return this->q_; };
            this->hess_ = [this](const vectorx_t& x) { return matrixx_t::Zero(this->dim_, this->dim_); };
        }

        vectorx_t GetCoefficients() { return this->q_; }
    private:
        vectorx_t q_; // the coefficients of the linear fn
    };
} // namespace torc::fn


#endif //TORC_LINEAR_FN_H
