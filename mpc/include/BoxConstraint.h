//
// Created by zolkin on 1/18/25.
//

#ifndef BOXCONSTRAINT_H
#define BOXCONSTRAINT_H
#include "constraint.h"


namespace torc::mpc {
    class BoxConstraint : public Constraint{
    public:
        BoxConstraint(int first_node, int last_node, const std::string& name, const vectorx_t& lb, const vectorx_t& ub,
            const std::vector<int>& idxs);

        void SetLowerBound(const vectorx_t& lb);
        void SetUpperBound(const vectorx_t& ub);
        void SetIdxs(const std::vector<int>& idxs);

        vectorx_t GetLowerBound(const vectorx_t& x_lin) const;
        vectorx_t GetUpperBound(const vectorx_t& x_lin) const;
        const std::vector<int>& GetIdxs() const;
    protected:
    private:
        vectorx_t lb_;
        vectorx_t ub_;
        std::vector<int> idxs_;
    };
}


#endif //BOXCONSTRAINT_H
