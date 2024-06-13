#ifndef TORC_CONSTRAINT_H
#define TORC_CONSTRAINT_H

#include <eigen3/Eigen/Dense>
#include "explicit_fn.h"
#include <string>
#include <ranges>


namespace torc::constraint {
    enum CONSTRAINT_TYPE {EQ, LEQ, GEQ};

    CONSTRAINT_TYPE reverse_type(CONSTRAINT_TYPE& ct) {
        return (ct == EQ) ? EQ : (ct == LEQ) ? GEQ : LEQ;
    }

    template <class scalar_t>
    class Constraint {
        using matrixx_t = Eigen::MatrixX<scalar_t>;
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using rowvecx_t = Eigen::RowVectorX<scalar_t>;

    public:
        Constraint(const std::vector<fn::ExplicitFn<scalar_t>>& functions,
                   const std::vector<scalar_t>& bounds,
                   const std::vector<CONSTRAINT_TYPE>& constraint_types,
                   const std::string& name="ConstraintInstance") {
            this->functions_ = functions;
            this->bounds_ = bounds;
            this->constraint_types_ = constraint_types;
            this->name_ = name;
        }

        bool Check(const vectorx_t& x) {
            return true;
        }

        void AddConstraint(const fn::ExplicitFn<scalar_t>& fn,
                           const scalar_t& bound,
                           const CONSTRAINT_TYPE& constraint_type) {
            this->functions_.push_back(fn);
            this->bounds_.push_back(bound);
            this->constraint_types_(constraint_type);
        }

        /**
         * Linearizes the constraints at the point x, into the form Ax <= b
         * @param x the point at which to linearize
         * @param A a matrix to hold the transposed gradients of the functions
         * @param upper_bound a vector to hold the upper bounds
         */
        void UnilateralForm(const vectorx_t& x, matrixx_t& A, vectorx_t& upper_bound) {
            std::vector<CONSTRAINT_TYPE> c_types = this->constraint_types_;
            size_t n_rows = this->functions_.size() + std::count(c_types.cbegin(), c_types.cend(), EQ);
            A.resize(n_rows, x.size());
            upper_bound.resize(n_rows);
            for (int i=0, i_row=0; i < this->functions_.size(); i++, i_row++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_(i) - fn(x);
                rowvecx_t grad_t = fn.Gradient(x).transposeInPlace();
                switch (c_types.at(i)) {
                    case EQ:
                        A.row(i_row) = grad_t;
                        upper_bound(i_row++) = bound;
                        A.row(i_row) = grad_t * -1.;
                        upper_bound(i_row) = bound * -1.;
                        break;
                    case LEQ:
                        A.row(i_row) = grad_t;
                        upper_bound(i_row) = bound;
                        break;
                    case GEQ:
                        A.row(i_row) = grad_t * -1.;
                        upper_bound(i_row) = bound * -1.;
                        break;
                    default:
                        break;
                }
            }
        }

        std::tuple<matrixx_t, vectorx_t> SparseUnilateralForm(const vectorx_t& x) {
            return std::make_tuple(1, 1);
        }

        void BoxForm(const vectorx_t& x, matrixx_t& A, vectorx_t& lower_bound, vectorx_t& upper_bound) {
            scalar_t min = std::numeric_limits<scalar_t>::min();
            scalar_t max = std::numeric_limits<scalar_t>::max();
            size_t n_rows = this->functions_.size();
            A.resize(n_rows, x.size());
            upper_bound.resize(n_rows);
            lower_bound.resize(n_rows);
            for (int i=0; i < n_rows; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_(i) - fn(x);
                A.row(i) = fn.Gradient(x).transposeInPlace();
                switch (this->constraint_types_.at(i)) {
                    case EQ:
                        upper_bound(i) = bound;
                        lower_bound(i) = bound;
                        break;
                    case LEQ:
                        upper_bound(i) = bound;
                        lower_bound(i) = min;
                        break;
                    case GEQ:
                        upper_bound(i) = max;
                        lower_bound(i) = bound;
                        break;
                    default:
                        break;
                }
            }
        }

        std::tuple<vectorx_t, matrixx_t, vectorx_t> SparseBoxForm(const vectorx_t& x) {
            return std::make_tuple(1, 1, 1);
        }

        void InequalityEqualityForm(const vectorx_t& x,
                                    matrixx_t& A,
                                    vectorx_t& upper_bound,
                                    matrixx_t& G,
                                    vectorx_t& equality_bound) {
            std::vector<CONSTRAINT_TYPE> c_types = this->constraint_types_;
            size_t n_functions = this->functions_.size();
            size_t G_n_rows = n_functions - std::count(c_types.cbegin(), c_types.cend(), EQ);
            size_t A_n_rows = n_functions - G_n_rows;
            A.resize(A_n_rows, x.size());
            G.resize(G_n_rows, x.size());
            upper_bound.resize(A_n_rows);
            equality_bound.resize(G_n_rows);

            for (int i=0, ineq_row=0, eq_row=0; i < this->functions_.size(); i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_(i) - fn(x);
                rowvecx_t grad = fn.Gradient(x).transposeInPlace();
                switch (c_types.at(i)) {
                    case EQ:
                        G.row(eq_row) = grad;
                        equality_bound.row(eq_row++) = grad;
                        break;
                    case LEQ:
                        A.row(ineq_row) = grad;
                        upper_bound(ineq_row++) = bound;
                        break;
                    case GEQ:
                        A.row(ineq_row) = grad * -1.;
                        upper_bound(ineq_row++) = bound * -1.;
                        break;
                    default:
                        break;
                }
            }
        }

        std::tuple<vectorx_t, matrixx_t, vectorx_t> SparseInequalityEqualityForm(const vectorx_t& x) {
            return std::make_tuple(1, 1, 1);
        }

    private:
        std::vector<fn::ExplicitFn<scalar_t>>& functions_;
        std::vector<scalar_t> bounds_;
        std::vector<CONSTRAINT_TYPE>& constraint_types_;
        std::string name_;
    };
}

#endif //TORC_CONSTRAINT_H
