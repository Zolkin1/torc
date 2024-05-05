#ifndef TORC_BASE_MODEL_H
#define TORC_BASE_MODEL_H

#include <eigen3/Eigen/Dense>
#include <utility>

namespace torc::models {
    class BaseModel {
    public:
        BaseModel(std::string name);

        std::string GetName() const;
    protected:
        std::string name_;
    private:
    };
} // torc::models

#endif //TORC_BASE_MODEL_H
