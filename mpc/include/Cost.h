//
// Created by zolkin on 1/20/25.
//

#ifndef COST_H
#define COST_H


#include <string>
#include <Eigen/Core>

namespace torc::mpc {
    using vectorx_t = Eigen::VectorXd;
    using matrixx_t = Eigen::MatrixXd;

    class Cost {
    public:
        Cost(int first_node, int last_node, const std::string &name, const vectorx_t& weights);

        int GetFirstNode() const;
        int GetLastNode() const;
        std::string GetName() const;

        virtual bool IsInNodeRange(int node) const;

        vectorx_t GetWeights() const;
    protected:
        int first_node_;
        int last_node_;
        std::string name_;

        vectorx_t weights_;
    private:
    };
}


#endif //COST_H
