//
// Created by zolkin on 1/18/25.
//

#ifndef INPUTCONSTRAINTS_H
#define INPUTCONSTRAINTS_H
#include <filesystem>
namespace fs = std::filesystem;

#include "constraint.h"


namespace torc::mpc {
    class FrictionConeConstraint : public Constraint {
    public:
        FrictionConeConstraint(int first_node, int last_node, const std::string& name, double friction_coef,
            double friction_margin, const fs::path& deriv_lib_path, bool compile_derivs);

        int GetNumConstraints() const override;

        std::pair<matrixx_t, vectorx_t> GetLinearization(const vectorx_t& f_lin) const;

        vectorx_t GetViolation(const vectorx_t& F, const vectorx_t& dF);

        static constexpr int CONE_SIZE = 4;
    protected:
    private:
        void ConeConstraint(const ad::ad_vector_t& df, const ad::ad_vector_t& fk_margin_coef, ad::ad_vector_t& violation) const;

        std::unique_ptr<ad::CppADInterface> constraint_function_;
        double friction_coef_;
        double friction_margin_;

        static constexpr int CONTACT_3DOF = 3;
    };
}


#endif //INPUTCONSTRAINTS_H
