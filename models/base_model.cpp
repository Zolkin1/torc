#include "base_model.h"

namespace torc::models {
    BaseModel::BaseModel(std::string name)
        : name_(std::move(name)) {}

    std::string BaseModel::GetName() const  { return name_; }

} // torc::models