//
// Created by zolkin on 8/21/24.
//

#ifndef VECTOR_AUTODIFF_FN_H
#define VECTOR_AUTODIFF_FN_H
#include <functional>
#include <Eigen/Core>

namespace torc::fn {
    namespace ADCG = CppAD::cg;
    namespace AD = CppAD;
    namespace fs = std::filesystem;

    using vectorx_t = Eigen::VectorX<scalar_t>;
    using matrixx_t = Eigen::MatrixX<scalar_t>;
    using cg_t = ADCG::CG<scalar_t>;
    using adcg_t = CppAD::AD<cg_t>;

    // TODO: Check if Forward(0, x) (i.e. zero order forward) is as quick as evaluating the double function
    class VectorAutodiffFn {
    public:
        VectorAutodiffFn(const std::function<adcg_t(const Eigen::VectorX<adcg_t>&)>& cg_fn,
                        const bool& force_generate=false,
                        const bool& timestamp_files=false,
                        const std::string& identifier="VectorAutodiffFnInstance") {
            name_ = identifier;

        }
    protected:
    private:
        std::string name_;
    };

} // namepsace torc::fn

#endif //VECTOR_AUTODIFF_FN_H
