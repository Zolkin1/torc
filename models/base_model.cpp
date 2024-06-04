#include "base_model.h"

namespace torc::models {
    BaseModel::BaseModel(std::string name)
        : name_(std::move(name)) {}

    std::string BaseModel::GetName() const  { return name_; }

    SystemType BaseModel::GetSystemType() const { return system_type_; }

} // torc::models