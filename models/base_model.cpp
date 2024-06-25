#include "base_model.h"

namespace torc::models {
    BaseModel::BaseModel(std::string name)
        : name_(std::move(name)) {}

    BaseModel::BaseModel(const torc::models::BaseModel& other) {
        name_ = other.name_;
        system_type_ = other.system_type_;
    }

    std::string BaseModel::GetName() const  { return name_; }

    SystemType BaseModel::GetSystemType() const { return system_type_; }

} // torc::models