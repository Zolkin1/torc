message(STATUS "Building models")

set(MODELS_INC "${torc_SOURCE_DIR}/models/include")
set(MODELS_HEADER_LIST
        "${MODELS_INC}/base_model.h"
        "${MODELS_INC}/robot_contact_info.h"
        "${MODELS_INC}/contact_model.h")

set(MODELS_SOURCES_LIST
    base_model.cpp
)

#----------- Pinocchio -----------#
if (BUILD_WITH_PINOCCHIO)
    message(STATUS "Building with Pinocchio")

    list(APPEND MODELS_SOURCES_LIST
                pinocchio_model.cpp
                full_order_rigid_body.cpp
                single_rigid_body.cpp)

    list(APPEND MODELS_HEADER_LIST
            "${MODELS_INC}/pinocchio_model.h"
            "${MODELS_INC}/full_order_rigid_body.h"
            "${MODELS_INC}/frame_state_types.h"
            "${MODELS_INC}/single_rigid_body.h")
endif ()

# -------  Create Library -------- #
add_library(Models
        ${MODELS_SOURCES_LIST}
        ${MODELS_HEADER_LIST})

target_include_directories(Models PUBLIC
        $<BUILD_INTERFACE:${MODELS_INC}>
        $<INSTALL_INTERFACE:models/include>)

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)
target_link_libraries(Models PUBLIC Eigen3::Eigen)

#----------- Pinocchio -----------#
if (BUILD_WITH_PINOCCHIO)
    find_package(pinocchio 3.0.0 REQUIRED)
    target_link_libraries(Models PUBLIC pinocchio::pinocchio)
endif ()