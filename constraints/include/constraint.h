#ifndef TORC_CONSTRAINT_H
#define TORC_CONSTRAINT_H

#include <eigen3/Eigen/Dense>
#include <string>
#include "explicit_fn.h"
#include "sparse_matrix_builder.h"

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
        using rowvecx_t = Eigen::VectorX<scalar_t>;
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

        void RawForm(const vectorx_t& x,
                     matrixx_t& A,
                     vectorx_t& bounds,
                     Eigen::VectorX<CONSTRAINT_T>& constraints) {
            if (this->constraint_types_.size() == 0) {
                return;
            }
            std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            size_t n_rows = this->functions_.size();
            A.resize(n_rows, x.size());
            constraints = this->constraint_types_;
            bounds.resize(n_rows);
            for (int i=0; i < n_rows; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                A.row(i) = fn.Gradient(x).transposeInPlace();
                bounds(i)= this->bounds_(i) - fn(x);
            }
        }

        void CompactRawForm(const vectorx_t& x,
                            matrixx_t& A,
                            vectorx_t& bounds,
                            Eigen::VectorX<CONSTRAINT_T>& constraints,
                            vectorx_t& annotations) {
            size_t n_rows = this->constraint_types_.size();
            if (n_rows == 0) {
                return;
            }
            std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            std::vector<CONSTRAINT_T> constraints_ = {};
            std::vector<size_t> annotations_ = {};

            n_rows = this->functions_.size();
            A.resize(n_rows, x.size());
            bounds.resize(n_rows);

            CONSTRAINT_T prev_type = this->constraint_types_.at(0);
            constraints.resize(n_rows);
            annotations = vectorx_t::Zero(n_rows);

            int row_compact = 0;

            for (int i=0; i < n_rows; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                A.row(i) = fn.Gradient(x).transposeInPlace();
                bounds(i) = this->bounds_(i) - fn(x);

                CONSTRAINT_T curr_type = this->constraint_types_.at(i);
                row_compact += (curr_type != prev_type);
                ++annotations(row_compact);
                constraints(row_compact) = curr_type;
            }
        }

        /**
         * Linearizes the constraints at the point x, into the form Ax <= b
         * @param x the point at which to linearize
         * @param A a matrix to hold the transposed gradients of the functions
         * @param upper_bound a vector to hold the upper bounds
         */
        void UnilateralForm(const vectorx_t& x, matrixx_t& A, vectorx_t& upper_bound) {
            if (this->constraint_types_.size() == 0) {
                return;
            }
            std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
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

        void SparseUnilateralForm(const vectorx_t& x, sp_matrix_t& A, vectorx_t& upper_bound) {
            if (this->constraint_types_.size() == 0) {
                return;
            }
            utils::SparseMatrixBuilder sp_builder = utils::SparseMatrixBuilder();
            std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            size_t n_rows = this->functions_.size() + std::count(c_types.cbegin(), c_types.cend(), EQ);
            upper_bound.resize(n_rows);

            for (int i=0, i_row=0; i < this->functions_.size(); i++, i_row++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_(i) - fn(x);
                rowvecx_t grad_t = fn.Gradient(x).transposeInPlace();
                switch (c_types.at(i)) {
                    case EQ:
                        sp_builder.SetMatrix(grad_t, i_row, 0);
                        upper_bound(i_row++) = bound;
                        sp_builder.SetMatrix(grad_t * -1, i_row, 0);
                        upper_bound(i_row) = bound * -1.;
                        break;
                    case LEQ:
                        sp_builder.SetMatrix(grad_t, i_row, 0);
                        upper_bound(i_row) = bound;
                        break;
                    case GEQ:
                        sp_builder.SetMatrix(grad_t * -1, i_row, 0);
                        upper_bound(i_row) = bound * -1.;
                        break;
                    default:
                        break;
                }
                A.setFromTriplets(sp_builder.GetTriplet().cbegin(), sp_builder.GetTriplet().cend());
            }
        }

        void BoxForm(const vectorx_t& x,
                     matrixx_t& A,
                     vectorx_t& lower_bound,
                     vectorx_t& upper_bound) {
            if (this->constraint_types_.size() == 0) {
                return;
            }
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


        void SparseBoxForm(const vectorx_t& x,
                           sp_matrix_t& A,
                           vectorx_t& lower_bound,
                           vectorx_t& upper_bound) {
            if (this->constraint_types_.size() == 0) {
                return;
            }
            const scalar_t min = std::numeric_limits<scalar_t>::min();
            const scalar_t max = std::numeric_limits<scalar_t>::max();
            auto sp_builder = utils::SparseMatrixBuilder();
            size_t n_rows = this->functions_.size();
            A.resize(n_rows, x.size());
            upper_bound.resize(n_rows);
            lower_bound.resize(n_rows);
            for (int i=0; i < n_rows; i++) {
                fn::ExplicitFn<scalar_t> fn = this->functions_.at(i);
                scalar_t bound = this->bounds_(i) - fn(x);
                sp_builder.SetMatrix(fn.Gradient(x).transposeInPlace(), i, 0);
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

        void InequalityEqualityForm(const vectorx_t& x,
                                    matrixx_t& A,
                                    vectorx_t& upper_bound,
                                    matrixx_t& G,
                                    vectorx_t& equality_bound) {
            const std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            if (c_types.empty()) {
                return;
            }
            const size_t n_functions = c_types.size();
            const size_t n_eq = std::count(c_types.cbegin(), c_types.cend(), EQ);
            const size_t n_ineq = n_functions - n_eq;

            A.resize(n_ineq, x.size());
            G.resize(n_eq, x.size());
            upper_bound.resize(n_ineq);
            equality_bound.resize(n_eq);

            for (int i=0, ineq_row=0, eq_row=0; i < n_functions; i++) {
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


        void SparseInequalityEqualityForm(const vectorx_t& x,
                                          sp_matrix_t& A,
                                          vectorx_t& upper_bound,
                                          sp_matrix_t& G,
                                          vectorx_t& equality_bound) {
            const std::vector<CONSTRAINT_T> c_types = this->constraint_types_;
            if (c_types.empty()) {
                return;
            }
            auto sp_builderA = utils::SparseMatrixBuilder();
            auto sp_builderG = utils::SparseMatrixBuilder();

            const size_t n_functions = c_types.size();
            const size_t n_eq = std::count(c_types.cbegin(), c_types.cend(), EQ);
            const size_t n_ineq = n_functions - n_eq;

            upper_bound.resize(n_ineq);
            equality_bound.resize(n_eq);

            for (int i=0, ineq_row=0, eq_row=0; i < n_functions; i++) {
                auto fn = this->functions_.at(i);
                scalar_t bound = this->bounds_(i) - fn(x);
                rowvecx_t grad = fn.Gradient(x).transposeInPlace();
                switch (c_types.at(i)) {
                    case EQ:
                        sp_builderG.SetMatrix(grad, eq_row, 0);
                        equality_bound.row(eq_row++) = grad;
                        break;
                    case LEQ:
                        sp_builderA.SetMatrix(grad, ineq_row, 0);
                        upper_bound(ineq_row++) = bound;
                        break;
                    case GEQ:
                        sp_builderA.SetMatrix(grad * -1., ineq_row, 0);
                        upper_bound(ineq_row++) = bound * -1.;
                        break;
                    default:
                        break;
                }
            }
            std::vector<Eigen::Triplet<scalar_t>> a_triplets = sp_builderA.GetTriplet();
            std::vector<Eigen::Triplet<scalar_t>> g_triplets = sp_builderG.GetTriplet();
            A.setFromTriplets(a_triplets.cbegin(), a_triplets.cend());
            G.setFromTriplets(g_triplets.cbegin(), g_triplets.cend());
        }



    private:
        std::vector<fn::ExplicitFn<scalar_t>> functions_;
        std::vector<scalar_t> bounds_;
        std::vector<CONSTRAINT_T> constraint_types_;
        std::string name_;
    };
}

#endif //TORC_CONSTRAINT_H
