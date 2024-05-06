#ifndef TORC_AUTODIFF_COST_H
#define TORC_AUTODIFF_COST_H

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cppad/cg.hpp>
#include "base_cost.h"

namespace torc {
    namespace ADCG = CppAD::cg;
    namespace AD = CppAD;

    /**
     * Class implementation of an arbitrary function, with auto-differentiation functionalities.
     * @tparam scalar_t the type of scalar used for the cost
     */
    template <class scalar_t>
    class AutodiffCost: public BaseCost<scalar_t> {
        using cg_t = ADCG::CG<scalar_t>;        // CodeGen scalar
        using adcg_t = CppAD::AD<cg_t>;         // CppAD scalar templated by CodeGen scalar
        using vectorx_t = Eigen::VectorX<scalar_t>;
        using matrixx_t = Eigen::MatrixX<scalar_t>;

    public:
        // There's no avoiding some leaking of the AD interface, because the function the user provides must return
        // an AD scalar and not a built-in one. Otherwise, the information used for differentiation stored in the result
        // is lost. Best case scenario, the user just needs to replace the function declaration and nothing else.
        /**
         * Constructor for the AutodiffCost class.
         * @param fn the cost function
         * @param dim the input dimension
         * @param identifier string identifier for the cost
         */
        AutodiffCost(const std::function<adcg_t(Eigen::VectorX<adcg_t>)>& fn,
                     const size_t dim=0,
                     const std::string& identifier="identifier") {
            this->fn_ = fn;
            this->dim_ = dim;
            this->identifier_ = identifier;

            std::vector<adcg_t> x(dim);
            CppAD::Independent(x);
            Eigen::VectorX<adcg_t> eigen_x = Eigen::Map<Eigen::VectorX<adcg_t> , Eigen::Unaligned>(x.data(), x.size());
            std::vector<adcg_t> y = {fn(eigen_x)};
            AD::ADFun<cg_t> ad_fn_(x, y);

            ADCG::ModelCSourceGen<double> cgen(ad_fn_, this->identifier_);
            cgen.setCreateJacobian(true);
            cgen.setCreateHessian(true);
            ADCG::ModelLibraryCSourceGen<double> libcgen(cgen);

            // compile source code
            ADCG::DynamicModelLibraryProcessor<double> p(libcgen);
            ADCG::GccCompiler<double> compiler;
            this->cg_dynamic_lib_ = p.createDynamicLibrary(compiler);

            // save to files
            ADCG::SaveFilesModelLibraryProcessor<double> p2(libcgen);
            p2.saveSources();
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
            adcg_t res = this->fn_(x_eigen);
            scalar_t result = AD::Value(res).getValue();    // first get the cg_t, then extract the scalar_t
            return result;
        }

        /**
         * Evaluates the gradient of the cost evaluated at x
         * @param x the input
         * @return grad f(x)
         */
        vectorx_t Gradient(const vectorx_t& x) const {
            std::unique_ptr<ADCG::GenericModel<double>> model = this->cg_dynamic_lib_->model(this->identifier_);
            std::vector<scalar_t> x_std(x.data(), x.data() + x.size());
            if (model->isJacobianAvailable()) {
                std::vector<scalar_t> jac = model->Jacobian(x_std);
                vectorx_t grad_eigen = Eigen::Map<vectorx_t , Eigen::Unaligned>(jac.data(), jac.size());
                return grad_eigen;
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
            std::unique_ptr<ADCG::GenericModel<double>> model = this->cg_dynamic_lib_->model(this->identifier_);
            std::vector<scalar_t> x_std(x.data(), x.data() + x.size());
            if (model->isHessianAvailable()) {
                std::vector<scalar_t> hess = model->Hessian(x_std, 0);
                matrixx_t grad_eigen(this->dim_, this->dim_);
                for (size_t row=0; row<this->dim_; row++) {
                    Eigen::RowVectorX<scalar_t> grad_row_eigen = Eigen::Map<vectorx_t , Eigen::Unaligned>(hess.data() + row*this->dim_, (row+1)*this->dim_);
                    grad_row_eigen.conservativeResize(this->dim_);  // so row assignment doesn't complain
                    grad_eigen.row(row) << grad_row_eigen;
                }
                return grad_eigen;
            } else {
                throw std::runtime_error("Hessian not available.");
            }
        }

    private:
        // the original function
        std::function<adcg_t(Eigen::VectorX<adcg_t>)> fn_;
        // the dynamic library that stores the differential information
        std::unique_ptr<ADCG::DynamicLib<double>> cg_dynamic_lib_;
    };
} // namespace torc

#endif //TORC_AUTODIFF_COST_H