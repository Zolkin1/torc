#include "linear_cost.h"

namespace torc {
    template <class dtype>
    LinearCost<dtype>::LinearCost(const Eigen::VectorX<dtype> &coefficients, const std::string &identifier) {
        this->coefficients = coefficients;
        this->identifier = identifier;
        this->domain_dim = coefficients.size();
    }

    template <class dtype>
    dtype LinearCost<dtype>::Evaluate(const Eigen::VectorX<dtype> &x) const {
        return this->coefficients.dot(x);
    }

    template <class dtype>
    Eigen::VectorX<dtype> LinearCost<dtype>::Gradient(const Eigen::VectorX<dtype> &x) const {
        return this->coefficients;
    }

    template <class dtype>
    Eigen::VectorX<dtype> LinearCost<dtype>::Gradient() const {
        return this->coefficients;
    }

    template <class dtype>
    Eigen::MatrixX<dtype> LinearCost<dtype>::Hessian(const Eigen::VectorX<dtype> &x) const {
        return Eigen::MatrixX<dtype>::Zero(this->domain_dim, this->domain_dim);
    }

    template<class dtype>
    Eigen::VectorX<dtype> LinearCost<dtype>::GetCoefficients() const {
        return this->coefficients;
    }
} // namespace torc
