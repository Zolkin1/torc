#ifndef TORC_AUTODIFF_COST_H
#define TORC_AUTODIFF_COST_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cppad/cg.hpp>
#include "base_cost.h"

namespace torc::cost {
    namespace ADCG = CppAD::cg;
    namespace AD = CppAD;

    /**
     * Class implementation of an arbitrary function, with auto-differentiation functionalities.
     * @tparam scalar_t the type of scalar used for the cost
     */
    template <class scalar_t>
    class AutodiffCost: public BaseCost<scalar_t> {
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;
        using cg_t = ADCG::CG<scalar_t>;
        using adcg_t = CppAD::AD<cg_t>;

    public:
        /**
         * Constructor for the AutodiffCost class.
         * @param fn the cost function
         * @param dim the input dimension
         * @param identifier string identifier for the cost
         */
        explicit AutodiffCost(const std::function<adcg_t(Eigen::VectorX<adcg_t>)>& fn,
                             const size_t dim=0,
                             const std::string& identifier="AutodiffCostInstance") {
            this->fn_ = fn;
            this->dim_ = dim;
            // the library has some issue with the identifier if it contains spaces/special characters. We impose a
            // more strict requirement on the identifier; strings like "-auto" will also work.
            if (this->IsValidIdentifier(identifier)) {
                this->identifier_ = identifier;
            } else {
                throw std::runtime_error("Identifier must be a valid variable name.");
            }

            // Record operations in the ADFun object
            std::vector<adcg_t> x(dim);
            CppAD::Independent(x);
            Eigen::VectorX<adcg_t> eigen_x = Eigen::Map<Eigen::VectorX<adcg_t> , Eigen::Unaligned>(x.data(), x.size());
            std::vector<adcg_t> y = {fn(eigen_x)};
            AD::ADFun<cg_t> ad_fn_(x, y);

            // Generate library source code
            ADCG::ModelCSourceGen<double> cgen(ad_fn_, this->identifier_);
            cgen.setCreateJacobian(true);
            cgen.setCreateHessian(true);
            ADCG::ModelLibraryCSourceGen<double> libcgen(cgen);

            // Compile source code
            ADCG::DynamicModelLibraryProcessor<double> p(libcgen);
            ADCG::GccCompiler<double> compiler;
            this->cg_dynamic_lib_ = p.createDynamicLibrary(compiler);
            this->cg_model_ = cg_dynamic_lib_->model(this->identifier_);
        }

        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return f(x)
         */
        scalar_t Evaluate(const vectorx_t& x) const {
            Eigen::VectorX<adcg_t> x_eigen(this->dim_);
            for (int i = 0; i < this->dim_; ++i) {
                x_eigen[i] = adcg_t(x[i]);
            }
            return AD::Value(this->fn_(x_eigen)).getValue();    // first get the cg_t, then extract the scalar_t
        }

        /**
         * Evaluates the gradient of the cost evaluated at x
         * @param x the input
         * @return grad f(x)
         */
        vectorx_t Gradient(const vectorx_t& x) const {
            const std::vector<scalar_t> x_std(x.data(), x.data() + x.size());
            if (cg_model_->isJacobianAvailable()) {
                std::vector<scalar_t> jac = cg_model_->Jacobian(x_std);
                return Eigen::Map<vectorx_t , Eigen::Unaligned>(jac.data(), jac.size());
            } else {
                throw std::runtime_error("Jacobian not available.");
            }
        }

        /**
         * Evaluates the Hessian of the cost evaluated at x
         * @param x the input
         * @return H_f(x)
         */
        matrixx_t Hessian(const vectorx_t& x) const {
            const std::vector<scalar_t> x_std(x.data(), x.data()+x.size());
            const int dim = this->dim_;
            if (cg_model_->isHessianAvailable()) {
                std::vector<scalar_t> hess = cg_model_->Hessian(x_std, 0);
                matrixx_t grad_eigen(dim, dim);
                for (size_t nrow=0; nrow < dim; nrow++) {
                    Eigen::RowVectorX<scalar_t> grad_row_eigen = Eigen::Map<Eigen::RowVectorX<scalar_t> , Eigen::Unaligned>(hess.data() + nrow * dim, (nrow + 1) * dim);
                    grad_row_eigen.conservativeResize(this->dim_);  // so nrow assignment doesn't complain
                    grad_eigen.row(nrow) << grad_row_eigen;
                }
                return grad_eigen;
            } else {
                throw std::runtime_error("Hessian not available.");
            }
        }

    private:
        std::function<adcg_t(Eigen::VectorX<adcg_t>)> fn_;          // the original function
        std::unique_ptr<ADCG::DynamicLib<double>> cg_dynamic_lib_;  // stores the operation tape and differential information
        std::unique_ptr<ADCG::GenericModel<double>> cg_model_;
    };
} // namespace torc::cost

#endif //TORC_AUTODIFF_COST_H