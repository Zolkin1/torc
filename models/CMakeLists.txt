message(STATUS "Building models")

set(MODELS_INC "${torc_SOURCE_DIR}/models/include")
set(MODELS_HEADER_LIST
        "${MODELS_INC}/base_model.h"
        "${MODELS_INC}/pinocchio_model.h"
        "${MODELS_INC}/rigid_body.h"
        "${MODELS_INC}/robot_state_types.h"
        "${MODELS_INC}/contact_state.h")

add_library(Models
        base_model.cpp
        pinocchio_model.cpp
        rigid_body.cpp
        contact_state.cpp
        ${MODELS_HEADER_LIST})

set_target_properties(Models PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Models PUBLIC
        $<BUILD_INTERFACE:${MODELS_INC}>
        $<INSTALL_INTERFACE:models/include>)

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)
target_link_libraries(Models PUBLIC Eigen3::Eigen)

#----------- Pinocchio -----------#
find_package(pinocchio 3.0.0 REQUIRED)
target_link_libraries(Models PUBLIC pinocchio::pinocchio)