#ifndef TORC_CONSTRAINT_H
#define TORC_CONSTRAINT_H

#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <string>
#include <iostream>
#include "explicit_fn.h"

#define CONSTRAINT_CHECK_EMPTY if (this->constraint_types_.empty()) { return; }

namespace torc::constraint {
    enum CONSTRAINT_T {
        EQ,     // equality constraint
        LEQ,    // lesser than or equal to constraint
        GEQ     // greater than or equal to constraint
    };

    /**
     * Represents a group of constraints, given in the form f(x) - constraint type - bound, and contains methods to
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
         * Default constructor.
         */
        explicit Constraint() {
            this->functions_ = std::vector<fn::ExplicitFn<scalar_t>>();
            this->bounds_ = std::vector<scalar_t>();
            this->constraint_types_ = std::vector<CONSTRAINT_T>();
            this->name_ = "ConstraintInstance";
        }

        /**
         * Constructor for the constraint class.
         * @param functions functions to be evaluated in the constraint
         * @param bounds the corresponding bounds of the functions
         * @param constraint_types relations that the functions must hold to the bounds (i.e. GEQ, LEQ, EQ)
         * @param name string identifier of the constraint
         */
        explicit Constraint(const std::vector<fn::ExplicitFn<scalar_t>>& functions,
                            const std::vector<scalar_t>& bounds,
                            const std::vector<CONSTRAINT_T>& constraint_types,
                            const std::string& name="ConstraintInstance") {
            this->functions_ = functions;
            this->bounds_ = bounds;
            this->constraint_types_ = constraint_types;
            this->name_ = name;
        }

        /**
         * Checks whether the constraint is satisfied at a given point
         * @param x the point to check the constraint
         * @return whether the constraint is satisfied
         */
        bool Check(const vectorx_t& x) {
            // iterate through all the constraints and check
            for (int i=0; i<this->functions_.size(); i++) {
                const fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                const scalar_t bound = this->bounds_.at(i);
                const CONSTRAINT_T type = this->constraint_types_.at(i);
                if (((type == GEQ) && (fn(x) < bound))
                    || ((type == LEQ) && (fn(x) > bound))
                    || ((type == EQ) && (fn(x) != bound))) {
                    return false;
                }
            }
            return true;
        }


        /**
         * Adds a single constraint to the constraint collection.
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
         * Computes the linearzation of all the constraints, and returns them in the form Ax <=/=/>= b
         * @param x the point to linearize the constraint functions
         * @param A the matrix comprised to be loaded from the transposed gradients of the function
         * @param bounds the vector to be loaded from the constraint bounds
         * @param constraint_types the std vector to be loaded from the constriant types
         */
        void RawForm(const vectorx_t& x,
                     matrixx_t& A,
                     vectorx_t& bounds,
                     std::vector<CONSTRAINT_T>& constraint_types) {
            CONSTRAINT_CHECK_EMPTY

            std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            size_t n_rows = this->functions_.size();
            A.resize(n_rows, x.size());
            constraint_types = this->constraint_types_;
            bounds.resize(n_rows);
            for (int i=0; i < n_rows; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                A.row(i) = fn.Gradient(x).transpose();
                bounds(i)= this->bounds_.at(i) - fn(x);
            }
        }


        /**
         *
         * @param x the point to lienarize the constraint functions
         * @param A the matrix to be loaded from the transposed gradients of the function
         * @param bounds the vector to be loaded from the constraint bounds
         * @param constraints the std vector to be loaded from the constraint types
         * @param annotations the number of repetitions that each constraint type appears in sequence
         */
        void CompactRawForm(const vectorx_t& x,
                            matrixx_t& A,
                            vectorx_t& bounds,
                            std::vector<CONSTRAINT_T>& constraints,
                            std::vector<size_t>& annotations) {
            CONSTRAINT_CHECK_EMPTY

            this->RawForm(x, A, bounds, constraints);

            CONSTRAINT_T prev_type = constraints.at(0);
            constraints.clear();
            annotations.clear();

            for (auto type : this->constraint_types_) {
                if ((type != prev_type) || annotations.empty()) {
                    constraints.push_back(type);
                    annotations.push_back(1);
                } else {
                    annotations.back()++;
                }
                prev_type = type;
            }
        }

        /**
         * Linearizes the constraints at the point x, into the form Ax <= b
         * @param x the point at which to linearize
         * @param A a matrix to hold the transposed gradients of the functions
         * @param upper_bound a vector to hold the upper bounds
         */
        void UnilateralForm(const vectorx_t& x, matrixx_t& A, vectorx_t& upper_bound) {
            CONSTRAINT_CHECK_EMPTY
            const std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            const size_t n_rows = this->functions_.size() + this->CountEq();

            A.resize(n_rows, x.size());
            upper_bound.resize(n_rows);
            for (int i=0, i_row=0; i < this->functions_.size(); i++, i_row++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_.at(i) - fn(x);
                rowvecx_t grad_t = fn.Gradient(x).transpose();
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


        /**
         * Linearizes the constraints at the point x, into the form Ax <= b, where A is sparse
         * @param x the point at which to linearize
         * @param A a sparse matrix to hold the transposed gradients of the functions
         * @param upper_bound a vector to hold the upper bounds
         */
        void SparseUnilateralForm(const vectorx_t& x, sp_matrix_t& A, vectorx_t& upper_bound) {
            CONSTRAINT_CHECK_EMPTY
            matrixx_t A_dense;
            this->UnilateralForm(x, A_dense, upper_bound);
            A = A_dense.sparseView();
        }


        /**
         * Linearizes the constraints at the point x, into the form lb <= Ax <= ub
         * @param x the point at which to linearize
         * @param A a matrix to hold the transposed gradients of the functions
         * @param lower_bound a vector to hold the lower bounds
         * @param upper_bound a vector to hold the upper bounds
         */
        void BoxForm(const vectorx_t& x,
                     matrixx_t& A,
                     vectorx_t& lower_bound,
                     vectorx_t& upper_bound) {
            CONSTRAINT_CHECK_EMPTY
            scalar_t max = std::numeric_limits<scalar_t>::max();
            scalar_t min = -max;
            size_t n_rows = this->functions_.size();
            A.resize(n_rows, x.size());
            upper_bound.resize(n_rows);
            lower_bound.resize(n_rows);
            for (int i=0; i < n_rows; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_.at(i) - fn(x);
                A.row(i) = fn.Gradient(x).transpose();
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


        /**
         * Linearizes the constraints at the point x, into the form lb <= Ax <= ub, where A is sparse
         * @param x the point at which to linearize
         * @param A a sparse matrix to hold the transposed gradients of the functions
         * @param lower_bound a vector to hold the lower bounds
         * @param upper_bound a vector to hold the upper bounds
         */
        void SparseBoxForm(const vectorx_t& x,
                           sp_matrix_t& A,
                           vectorx_t& lower_bound,
                           vectorx_t& upper_bound) {
            CONSTRAINT_CHECK_EMPTY
            matrixx_t A_dense;
            this->BoxForm(x, A_dense, lower_bound, upper_bound);
            A = A_dense.sparseView();
        }


        /**
         * Linearizes the constraints at the point x, into the form Ax <= ub, Gx = eb
         * @param x the point at which to linearizes
         * @param A a matrix to hold the transposed gradients of the functions in inequality constraints
         * @param upper_bound a vector to hold the upper bounds of the inequality constraints
         * @param G a matrix to hold the transposed gradients of the functions in equality constraints
         * @param equality_bound a vector to hold the lower bounds of the equality constraints
         */
        void InequalityEqualityForm(const vectorx_t& x,
                                    matrixx_t& A,
                                    vectorx_t& upper_bound,
                                    matrixx_t& G,
                                    vectorx_t& equality_bound) {
            CONSTRAINT_CHECK_EMPTY
            const std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            const size_t n_functions = c_types.size();
            const size_t n_eq = this->CountEq();
            const size_t n_ineq = n_functions - n_eq;

            A.resize(n_ineq, x.rows());
            G.resize(n_eq, x.rows());
            upper_bound.resize(n_ineq, 1);
            equality_bound.resize(n_eq, 1);

            for (int i=0, ineq_row=0, eq_row=0; i < n_functions; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_.at(i) - fn(x);
                rowvecx_t grad = fn.Gradient(x).transpose();
                switch (c_types.at(i)) {
                    case EQ:
                        G.row(eq_row) = grad;
                        equality_bound(eq_row++) = bound;
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


        /**
         * Linearizes the constraints at the point x, into the form Ax <= ub, Gx = eb, where A and G are sparse
         * @param x the point at which to linearizes
         * @param A a sparse matrix to hold the transposed gradients of the functions in inequality constraints
         * @param upper_bound a vector to hold the upper bounds of the inequality constraints
         * @param G a sparse matrix to hold the transposed gradients of the functions in equality constraints
         * @param equality_bound a vector to hold the lower bounds of the equality constraints
         */
        void SparseInequalityEqualityForm(const vectorx_t& x,
                                          sp_matrix_t& A,
                                          vectorx_t& upper_bound,
                                          sp_matrix_t& G,
                                          vectorx_t& equality_bound) {
            CONSTRAINT_CHECK_EMPTY
            matrixx_t A_dense, G_dense;
            this->InequalityEqualityForm(x, A_dense, upper_bound, G_dense, equality_bound);
            A = A_dense.sparseView();
            G = G_dense.sparseView();
        }


    private:
        std::vector<fn::ExplicitFn<scalar_t>> functions_;   // the functions to evaluate in the constraint
        std::vector<scalar_t> bounds_;      // the bounds of the functions in the constraint
        std::vector<CONSTRAINT_T> constraint_types_;    // the constraint types (i.e., LEQ, GEQ, EQ)
        std::string name_;  // string identifier

        [[nodiscard]] size_t CountEq() const {
            return std::count(constraint_types_.cbegin(), constraint_types_.cend(), EQ);
        }
    };
}

#endif //TORC_CONSTRAINT_H
