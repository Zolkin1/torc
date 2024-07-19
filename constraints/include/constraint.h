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
        Equals,         // equality constraint
        LessThan,       // less than or equal to constraint
        GreaterThan     // greater than or equal to constraint
    };

    using matrixx_t = Eigen::MatrixXd;
    using vectorx_t = Eigen::VectorXd;
    using rowvecx_t = Eigen::RowVectorXd;
    using sp_matrix_t = Eigen::SparseMatrix<double>;

    struct OriginalFormData {
        matrixx_t grads;
        vectorx_t bounds;
        std::vector<CONSTRAINT_T> types;
    };

    struct CompactOriginalFormData {
        matrixx_t grads;
        vectorx_t bounds;
        std::vector<CONSTRAINT_T> types;
        std::vector<size_t> reps;
    };

    struct UnilateralConstraints {
        matrixx_t grads;
        vectorx_t bounds;
    };

    struct SparseUnilateralConstraints {
        sp_matrix_t grads;
        vectorx_t bounds;
    };

    struct BoxConstraints {
        matrixx_t A;
        vectorx_t ub;
        vectorx_t lb;
    };

    struct SparseBoxConstraints {
        sp_matrix_t A;
        vectorx_t ub;
        vectorx_t lb;
    };

    struct InequalityEqualityConstraints {
        matrixx_t ineq_grad;
        matrixx_t eq_grad;
        vectorx_t ineq_bounds;
        vectorx_t eq_bounds;
    };

    struct SparseInequalityEqualityConstraints {
        sp_matrix_t ineq_grad;
        sp_matrix_t eq_grad;
        vectorx_t ineq_bounds;
        vectorx_t eq_bounds;
    };

    /**
     * @brief Represents a group of constraints, given in the form f(x) - constraint type - bound, and contains methods to
     * linearize the constraints in various forms.
     * @tparam scalar_t the scalar type for the constraint
     */
    template <class scalar_t>
    class Constraint {
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
         * @param eps floating point comparison limit
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
         * @param data struct to hold return values
         */
        void OriginalForm(const vectorx_t& x, OriginalFormData& data) {
            CONSTRAINT_CHECK_EMPTY
            std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            size_t n_rows = this->functions_.size();
            data.grads.resize(n_rows, x.size());
            data.types = this->constraint_types_;
            data.bounds.resize(n_rows);
            for (int i=0; i < n_rows; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                data.grads.row(i) = fn.Gradient(x).transpose();
                data.bounds(i)= this->bounds_.at(i) - fn(x);
            }
        }

        /**
         * @brief Computes the linearization of all the constraints, and returns them in the form Ax <=/=/>= b, in the
         * order that the user passed them in. The constraints vector is slightly modified, where if a number of
         * neighboring types are identical, they are represented with one entry in the constraints vector and an entry
         * in the annotations vector, which describes the number of repetitions of that constraint.
         * @param x the point to linearize the constraint functions
         * @param data struct to hold return values
         */
        void CompactOriginalForm(const vectorx_t& x, CompactOriginalFormData& data) {
            CONSTRAINT_CHECK_EMPTY
            OriginalFormData original_data;
            this->OriginalForm(x, original_data);

            data.grads = original_data.grads;
            data.bounds = original_data.bounds;
            data.types.clear();
            data.reps.clear();

            CONSTRAINT_T prev_type = original_data.types.at(0);
            for (auto type : this->constraint_types_) {
                if ((type != prev_type) || data.reps.empty()) {
                    data.types.push_back(type);
                    data.reps.push_back(1);
                } else {
                    ++data.reps.back();
                }
                prev_type = type;
            }
        }

        /**
         * @brief Linearizes the constraints at the point x, into the form Ax <= b
         * @param x the point at which to linearize
         * @param data struct to hold return values
         */
        void UnilateralForm(const vectorx_t& x, UnilateralConstraints& data) {
            CONSTRAINT_CHECK_EMPTY
            const std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            const size_t n_rows = this->functions_.size() + this->CountEq();

            data.grads.resize(n_rows, x.size());
            data.bounds.resize(n_rows);
            for (int i=0, i_row=0; i < this->functions_.size(); i++, i_row++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_.at(i) - fn(x);
                rowvecx_t grad_t = fn.Gradient(x).transpose();
                switch (c_types.at(i)) {
                    case Equals:
                        data.grads.row(i_row) = grad_t;
                        data.bounds(i_row++) = bound;
                        data.grads.row(i_row) = grad_t * -1.;
                        data.bounds(i_row) = bound * -1.;
                        break;
                    case LessThan:
                        data.grads.row(i_row) = grad_t;
                        data.bounds(i_row) = bound;
                        break;
                    case GreaterThan:
                        data.grads.row(i_row) = grad_t * -1.;
                        data.bounds(i_row) = bound * -1.;
                        break;
                    default:
                        break;
                }
            }
        }

        /**
         * @brief Linearizes the constraints at the point x, into the form Ax <= b, where A is sparse
         * @param x the point at which to linearize
         * @param data struct to hold return values
         */
        void SparseUnilateralForm(const vectorx_t& x,
                                  SparseUnilateralConstraints& data) {
            UnilateralConstraints dense_data;
            CONSTRAINT_CHECK_EMPTY

            this->UnilateralForm(x, dense_data);
            data.grads = dense_data.grads.sparseView();
            data.bounds = dense_data.bounds;
        }

        /**
         * @brief Linearizes the constraints at the point x, into the form lb <= Ax <= ub
         * @param x the point at which to linearize
         * @param data struct to hold return values
         */
        void BoxForm(const vectorx_t& x,
                     BoxConstraints& data) {
            CONSTRAINT_CHECK_EMPTY
            scalar_t max = std::numeric_limits<scalar_t>::max();
            scalar_t min = -max;
            size_t n_rows = this->functions_.size();
            data.A.resize(n_rows, x.size());
            data.ub.resize(n_rows);
            data.lb.resize(n_rows);
            for (int i=0; i < n_rows; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_.at(i) - fn(x);
                data.A.row(i) = fn.Gradient(x).transpose();
                switch (this->constraint_types_.at(i)) {
                    case Equals:
                        data.ub(i) = bound;
                        data.lb(i) = bound;
                        break;
                    case LessThan:
                        data.ub(i) = bound;
                        data.lb(i) = min;
                        break;
                    case GreaterThan:
                        data.ub(i) = max;
                        data.lb(i) = bound;
                        break;
                    default:
                        break;
                }
            }
        }

        /**
         * @brief Linearizes the constraints at the point x, into the form lb <= Ax <= ub, where A is sparse
         * @param x the point at which to linearize
         * @param data struct to hold return values
         */
        void SparseBoxForm(const vectorx_t& x,
                           SparseBoxConstraints& data) {
            CONSTRAINT_CHECK_EMPTY
            BoxConstraints dense_data;
            this->BoxForm(x, dense_data);
            data.A = dense_data.A.sparseView();
            data.ub = dense_data.ub;
            data.lb = dense_data.lb;
        }

        /**
         * @brief Linearizes the constraints at the point x, into the form Ax <= ub, Gx = eb
         * @param x the point at which to linearizes
         * @param data struct to hold return values
         */
        void InequalityEqualityForm(const vectorx_t& x,
                                    InequalityEqualityConstraints& data) {
            CONSTRAINT_CHECK_EMPTY
            const std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            const size_t n_functions = c_types.size();
            const size_t n_eq = this->CountEq();
            const size_t n_ineq = n_functions - n_eq;

            data.ineq_grad.resize(n_ineq, x.rows());
            data.eq_grad.resize(n_eq, x.rows());
            data.ineq_bounds.resize(n_ineq, 1);
            data.eq_bounds.resize(n_eq, 1);

            for (int i=0, ineq_row=0, eq_row=0; i < n_functions; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_.at(i) - fn(x);
                rowvecx_t grad = fn.Gradient(x).transpose();
                switch (c_types.at(i)) {
                    case Equals:
                        data.eq_grad.row(eq_row) = grad;
                        data.eq_bounds(eq_row++) = bound;
                        break;
                    case LessThan:
                        data.ineq_grad.row(ineq_row) = grad;
                        data.ineq_bounds(ineq_row++) = bound;
                        break;
                    case GreaterThan:
                        data.ineq_grad.row(ineq_row) = grad * -1.;
                        data.ineq_bounds(ineq_row++) = bound * -1.;
                        break;
                    default:
                        break;
                }
            }
        }

        /**
         * @brief Linearizes the constraints at the point x, into the form Ax <= ub, Gx = eb, where A and G are sparse
         * @param x the point at which to linearizes
         * @param data struct to hold return values
         */
        void SparseInequalityEqualityForm(const vectorx_t& x,
                                          SparseInequalityEqualityConstraints& data) {
            CONSTRAINT_CHECK_EMPTY
            InequalityEqualityConstraints dense_data;
            this->InequalityEqualityForm(x, dense_data);
            data.ineq_grad = dense_data.ineq_grad.sparseView();
            data.eq_grad = dense_data.eq_grad.sparseView();
            data.ineq_bounds = dense_data.ineq_bounds;
            data.eq_bounds = dense_data.eq_bounds;
        }


    private:
        std::vector<fn::ExplicitFn<scalar_t>> functions_;   // the functions to evaluate in the constraint
        std::vector<scalar_t> bounds_;                      // the bounds of the functions in the constraint
        std::vector<CONSTRAINT_T> constraint_types_;        // the constraint types (i.e., LEQ, GEQ, EQ)
        scalar_t eps_;                                      // floating point comparison accuracy
        std::string name_;                                  // string identifier

        [[nodiscard]] size_t CountEq() const {
            return std::count(constraint_types_.cbegin(), constraint_types_.cend(), Equals);
        }
    };
}

#endif //TORC_CONSTRAINT_H
