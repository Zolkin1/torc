#ifndef TORC_AUTODIFF_COST_H
#define TORC_AUTODIFF_COST_H

#include <vector>
#include "base_cost.h"
#include <cppad/cg.hpp>
#include <iostream>
#include <cppad/example/code_gen_fun.hpp>

namespace torc {
    /**
     * Class implementation of an arbitrary function, with auto-differentiation functionalities.
     * @tparam scalar_t the type of scalar used for the cost
     */
    namespace ADCG = CppAD::cg;
    namespace AD = CppAD;

    template <class scalar_t>
    class AutodiffCost: public BaseCost<scalar_t> {
        typedef ADCG::CG<scalar_t> cg_t;        // CodeGen scalar
        typedef CppAD::AD<cg_t> adcg_t;         // CppAD scalar templated by CodeGen scalar
        typedef Eigen::VectorX<scalar_t> vectorx_t;
        typedef Eigen::MatrixX<scalar_t> matrixx_t;

    public:
        // There's no avoiding some leaking of the AD interface, because the function the user provides must return
        // an AD scalar and not a built-in one. Otherwise, the information used for differentiation stored in the result
        // is lost. Best case scenario, the user just needs to replace the function declaration and nothing else.
        explicit AutodiffCost(const std::function<adcg_t(Eigen::VectorX<adcg_t>)>& fn,
                              const size_t dim=0,
                              const std::string& identifier="Auto-Differentiation Cost Instance") {
            this->fn_ = fn;
            this->dim_ = dim;
            this->identifier_ = identifier;

            std::vector<adcg_t> x(dim);
            CppAD::Independent(x);
            Eigen::VectorX<adcg_t> eigen_x;
            eigen_x.resize(dim);
            eigen_x(x.data());

            std::vector<adcg_t> y(1);
            y[0] = fn(eigen_x);

            AD::ADFun<cg_t> ad_fn_(x, y);       // this is templated by two classes?
            std::string filename = "example_code";
            code_gen_fun f(filename, ad_fn_);   // while this demands a template that has only one class.
        }

        /**
         * Evaluates the cost function at a given point
         * @param x the input to the function
         * @return f(x)
         */
        scalar_t Evaluate(const vectorx_t& x) const {
//            Eigen::VectorX<adcg_t> x_in;
//            x_in.resize(this->dim_);
//            for (int i = 0; i < this->dim_; ++i) {
//                adcg_t element(x[i]);
//                x_in[i] = element;
//            }
//            adcg_t res = this->fn_(x_in);
//            scalar_t result = ExtractADCGValue(res);
//            return result;
        }

        /**
         * Evaluates the gradient of the cost evaluated at x
         * @param x the input
         * @return grad f(x)
         */
        vectorx_t Gradient(const vectorx_t& x) const {
//            std::unique_ptr<ADCG::GenericModel<double>> model = cg_dynamic_lib_->model(this->identifier_);
//            std::vector<scalar_t> jac_vec(this->dim_);
//            std::vector<scalar_t> x_vec(x.data(), x.data() + x.size());
//            model->Jacobian(x_vec, jac_vec);
//            vectorx_t jac(jac_vec.data());
//            return jac;
        }

        /**
         * Evaluates the Hessian of the cost evaluated at x
         * @param x the input
         * @return H_f(x)
         */
        matrixx_t Hessian(const vectorx_t& x) const {
//            std::unique_ptr<ADCG::GenericModel<double>> model = cg_dynamic_lib_->model(this->identifier_);
//            std::vector<double> w(1, 1.0);
//            std::vector<double> hess;
//            std::vector<size_t> row, col;
//
//            model->SparseHessian(x, w, hess, row, col);
//
//            // print out the result
//            for (size_t i = 0; i < hess.size(); ++i)
//                std::cout << "(" << row[i] << "," << col[i] << ") " << hess[i] << std::endl;
        }

    private:
        std::function<adcg_t(Eigen::VectorX<adcg_t>)> fn_;
//        adfun_t ad_fn_;
        std::unique_ptr<ADCG::DynamicLib<double>> cg_dynamic_lib_;

        static scalar_t ExtractADCGValue(adcg_t x) { return AD::Value(x).getValue(); }
    };
} // namespace torc

#endif //TORC_AUTODIFF_COST_H