#include "base_model.h"

#include <utility>

namespace torc::models {

    BaseModel::BaseModel(std::string name,
                         const SystemType& system_type)
        : name_(std::move(name)), system_type_(system_type) {}

    BaseModel::BaseModel(const BaseModel& other) {
        name_ = other.name_;
        system_type_ = other.system_type_;
    }

    std::string BaseModel::GetName() const { return name_; }

    template <class... Args>
    void BaseModel::ParseState(const vectorx_t &state, Args... args) {
        // HandleParseState(state, {args...});
    }

    // void BaseModel::HandleParseState(const vectorx_t &state,
                                     // std::initializer_list<vectorx_t> args) {
        // assert(false && "BaseModel cannot parse states");
    // }

    SystemType BaseModel::GetSystemType() const { return system_type_; }

} // torc::models