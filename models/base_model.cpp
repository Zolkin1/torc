#include "base_model.h"

#include <utility>

namespace torc::models {

    BaseModel::BaseModel(const std::string& name,
                         const SystemType& system_type)
        : name_(name), system_type_(system_type) {}

    BaseModel::BaseModel(const BaseModel& other) {
        name_ = other.name_;
        system_type_ = other.system_type_;
    }

    std::string BaseModel::GetName() const { return name_; }

    SystemType BaseModel::GetSystemType() const { return system_type_; }

} // torc::models