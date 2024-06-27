#ifndef TORC_CONSTRAINT_H
#define TORC_CONSTRAINT_H

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <string>
#include <iostream>
#include "explicit_fn.h"
#include "constraint_data.h"

#define CONSTRAINT_CHECK_EMPTY if (this->constraint_types_.empty()) { return; }

namespace torc::constraint {

    /**
     * @brief Represents a group of constraints, given in the form f(x) - constraint type - bound, and contains methods to
     * linearize the constraints in various forms.
     * @tparam scalar_t the scalar type for the constraint
     */
    template <class scalar_t>
    class Constraint {
        using matrixx_t = Eigen::MatrixX<scalar_t>;
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using rowvecx_t = Eigen::RowVectorX<scalar_t>;
        using sp_matrix_t = Eigen::SparseMatrix<scalar_t>;

    public:
        /**
         * @brief Default constructor.
         */
        explicit Constraint() {
            this->functions_ = std::vector<fn::ExplicitFn<scalar_t>>();
            this->bounds_ = std::vector<scalar_t>();
            this->constraint_types_ = std::vector<CONSTRAINT_T>();
            this->name_ = "ConstraintInstance";
        }

        /**
         * @brief Constructor for the constraint class.
         * @param functions functions to be evaluated in the constraint
         * @param bounds the corresponding bounds of the functions
         * @param constraint_types relations that the functions must hold to the bounds (i.e. GEQ, LEQ, EQ)
         * @param name string identifier of the constraint
         */
        explicit Constraint(const std::vector<fn::ExplicitFn<scalar_t>>& functions,
                            const std::vector<scalar_t>& bounds,
                            const std::vector<CONSTRAINT_T>& constraint_types,
                            const scalar_t eps=1e-10,
                            const std::string& name="ConstraintInstance") {
            this->functions_ = functions;
            this->bounds_ = bounds;
            this->constraint_types_ = constraint_types;
            this->name_ = name;
            this->eps_ = eps;
        }

        /**
         * @brief Checks whether the constraint is satisfied at a given point
         * @param x the point to check the constraint
         * @return whether the constraint is satisfied
         */
        bool Check(const vectorx_t& x) {
            // iterate through all the constraints and check
            for (int i=0; i<this->functions_.size(); i++) {
                const fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                const scalar_t bound = this->bounds_.at(i);
                const CONSTRAINT_T type = this->constraint_types_.at(i);
                if (((type == GreaterThan) && (fn(x) < bound))
                    || ((type == LessThan) && (fn(x) > bound))
                    || ((type == Equals) && (std::abs(fn(x)-bound) > eps_))) {
                    return false;
                }
            }
            return true;
        }


        /**
         * @brief Adds a single constraint to the constraint collection.
         * @param fn the function to be evaluated in the constraint
         * @param bound the bound of the function
         * @param constraint_type LEQ, GEQ, EQ
         */
        void AddConstraint(const fn::ExplicitFn<scalar_t>& fn,
                           const scalar_t& bound,
                           const CONSTRAINT_T& constraint_type) {
            this->functions_.push_back(fn);
            this->bounds_.push_back(bound);
            this->constraint_types_.push_back(constraint_type);
        }

        /**
         * @brief Computes the linearzation of all the constraints, and returns them in the form Ax <=/=/>= b, in the
         * order that the user passed them in.
         * @param x the point to linearize the constraint functions
         * @param constraint_data ineq_grad, types, bound_all changed
         */
        void OriginalForm(const vectorx_t& x, ConstraintData<scalar_t>& constraint_data) {
            CONSTRAINT_CHECK_EMPTY
            std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            size_t n_rows = this->functions_.size();
            constraint_data.ineq_grad.resize(n_rows, x.size());
            constraint_data.types = this->constraint_types_;
            constraint_data.bound_all.resize(n_rows);
            for (int i=0; i < n_rows; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                constraint_data.ineq_grad.row(i) = fn.Gradient(x).transpose();
                constraint_data.bound_all(i)= this->bounds_.at(i) - fn(x);
            }
        }


        /**
         * @brief Computes the linearization of all the constraints, and returns them in the form Ax <=/=/>= b, in the
         * order that the user passed them in. The constraints vector is slightly modified, where if a number of
         * neighboring types are identical, they are represented with one entry in the constraints vector and an entry
         * in the annotations vector, which describes the number of repetitions of that constraint.
         * @param x the point to linearize the constraint functions
         * @param constraint_data types, reps, ineq_grad, bound_all changed
         */
        void CompactOriginalForm(const vectorx_t& x, ConstraintData<scalar_t>& constraint_data) {
            CONSTRAINT_CHECK_EMPTY
            this->OriginalForm(x, constraint_data);

            CONSTRAINT_T prev_type = constraint_data.types.at(0);
            constraint_data.types.clear();
            constraint_data.reps.clear();

            for (auto type : this->constraint_types_) {
                if ((type != prev_type) || constraint_data.reps.empty()) {
                    constraint_data.types.push_back(type);
                    constraint_data.reps.push_back(1);
                } else {
                    ++constraint_data.reps.back();
                }
                prev_type = type;
            }
        }

        /**
         * @brief Linearizes the constraints at the point x, into the form Ax <= b
         * @param x the point at which to linearize
         * @param constraint_data ineq_grad, bound_high changed
         */
        void UnilateralForm(const vectorx_t& x, ConstraintData<scalar_t>& constraint_data) {
            CONSTRAINT_CHECK_EMPTY
            const std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            const size_t n_rows = this->functions_.size() + this->CountEq();

            constraint_data.ineq_grad.resize(n_rows, x.size());
            constraint_data.bound_high.resize(n_rows);
            for (int i=0, i_row=0; i < this->functions_.size(); i++, i_row++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_.at(i) - fn(x);
                rowvecx_t grad_t = fn.Gradient(x).transpose();
                switch (c_types.at(i)) {
                    case Equals:
                        constraint_data.ineq_grad.row(i_row) = grad_t;
                        constraint_data.bound_high(i_row++) = bound;
                        constraint_data.ineq_grad.row(i_row) = grad_t * -1.;
                        constraint_data.bound_high(i_row) = bound * -1.;
                        break;
                    case LessThan:
                        constraint_data.ineq_grad.row(i_row) = grad_t;
                        constraint_data.bound_high(i_row) = bound;
                        break;
                    case GreaterThan:
                        constraint_data.ineq_grad.row(i_row) = grad_t * -1.;
                        constraint_data.bound_high(i_row) = bound * -1.;
                        break;
                    default:
                        break;
                }
            }
        }


        /**
         * @brief Linearizes the constraints at the point x, into the form Ax <= b, where A is sparse
         * @param x the point at which to linearize
         * @param constraint_data ineq_grad, bound_high, ineq_grad_sparse changed
         */
        void SparseUnilateralForm(const vectorx_t& x,
                                  ConstraintData<scalar_t>& constraint_data) {
            CONSTRAINT_CHECK_EMPTY
            this->UnilateralForm(x, constraint_data);
            constraint_data.ineq_grad_sparse = constraint_data.ineq_grad.sparseView();
        }


        /**
         * @brief Linearizes the constraints at the point x, into the form lb <= Ax <= ub
         * @param x the point at which to linearize
         * @param constraint_data ineq_grad, bound_high, bound_low changed
         */
        void BoxForm(const vectorx_t& x,
                     ConstraintData<scalar_t>& constraint_data) {
            CONSTRAINT_CHECK_EMPTY
            scalar_t max = std::numeric_limits<scalar_t>::max();
            scalar_t min = -max;
            size_t n_rows = this->functions_.size();
            constraint_data.ineq_grad.resize(n_rows, x.size());
            constraint_data.bound_high.resize(n_rows);
            constraint_data.bound_low.resize(n_rows);
            for (int i=0; i < n_rows; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_.at(i) - fn(x);
                constraint_data.ineq_grad.row(i) = fn.Gradient(x).transpose();
                switch (this->constraint_types_.at(i)) {
                    case Equals:
                        constraint_data.bound_high(i) = bound;
                        constraint_data.bound_low(i) = bound;
                        break;
                    case LessThan:
                        constraint_data.bound_high(i) = bound;
                        constraint_data.bound_low(i) = min;
                        break;
                    case GreaterThan:
                        constraint_data.bound_high(i) = max;
                        constraint_data.bound_low(i) = bound;
                        break;
                    default:
                        break;
                }
            }
        }


        /**
         * @brief Linearizes the constraints at the point x, into the form lb <= Ax <= ub, where A is sparse
         * @param x the point at which to linearize
         * @param constraint_data ineq_grad_sparse, ineq_grad, bound_high, bound_low changed
         */
        void SparseBoxForm(const vectorx_t& x,
                           ConstraintData<scalar_t>& constraint_data) {
            CONSTRAINT_CHECK_EMPTY
            this->BoxForm(x, constraint_data);
            constraint_data.ineq_grad_sparse = constraint_data.ineq_grad.sparseView();
        }


        /**
         * @brief Linearizes the constraints at the point x, into the form Ax <= ub, Gx = eb
         * @param x the point at which to linearizes
         * @param constraint_data ineq_grad, eq_grad, bound_high, bound_eq changed
         */
        void InequalityEqualityForm(const vectorx_t& x,
                                    ConstraintData<scalar_t>& constraint_data) {
            CONSTRAINT_CHECK_EMPTY
            const std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            const size_t n_functions = c_types.size();
            const size_t n_eq = this->CountEq();
            const size_t n_ineq = n_functions - n_eq;

            constraint_data.ineq_grad.resize(n_ineq, x.rows());
            constraint_data.eq_grad.resize(n_eq, x.rows());
            constraint_data.bound_high.resize(n_ineq, 1);
            constraint_data.bound_eq.resize(n_eq, 1);

            for (int i=0, ineq_row=0, eq_row=0; i < n_functions; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_.at(i) - fn(x);
                rowvecx_t grad = fn.Gradient(x).transpose();
                switch (c_types.at(i)) {
                    case Equals:
                        constraint_data.eq_grad.row(eq_row) = grad;
                        constraint_data.bound_eq(eq_row++) = bound;
                        break;
                    case LessThan:
                        constraint_data.ineq_grad.row(ineq_row) = grad;
                        constraint_data.bound_high(ineq_row++) = bound;
                        break;
                    case GreaterThan:
                        constraint_data.ineq_grad.row(ineq_row) = grad * -1.;
                        constraint_data.bound_high(ineq_row++) = bound * -1.;
                        break;
                    default:
                        break;
                }
            }
        }


        /**
         * @brief Linearizes the constraints at the point x, into the form Ax <= ub, Gx = eb, where A and G are sparse
         * @param x the point at which to linearizes
         * @param constraint_data ineq_grad_sparse, eq_grad_sparse, ineq_grad, eq_grad, bound_high, bound_eq changed
         */
        void SparseInequalityEqualityForm(const vectorx_t& x,
                                          ConstraintData<scalar_t>& constraint_data) {
            CONSTRAINT_CHECK_EMPTY
            matrixx_t A_dense, G_dense;
            this->InequalityEqualityForm(x, constraint_data);
            constraint_data.ineq_grad_sparse = constraint_data.ineq_grad.sparseView();
            constraint_data.eq_grad_sparse = constraint_data.eq_grad.sparseView();
        }


    private:
        std::vector<fn::ExplicitFn<scalar_t>> functions_;   // the functions to evaluate in the constraint
        std::vector<scalar_t> bounds_;      // the bounds of the functions in the constraint
        std::vector<CONSTRAINT_T> constraint_types_;    // the constraint types (i.e., LEQ, GEQ, EQ)
        scalar_t eps_;
        std::string name_;  // string identifier

        [[nodiscard]] size_t CountEq() const {
            return std::count(constraint_types_.cbegin(), constraint_types_.cend(), Equals);
        }
    };
}

#endif //TORC_CONSTRAINT_H
