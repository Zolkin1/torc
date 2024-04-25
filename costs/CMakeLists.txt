message(STATUS "Building costs")

set(COST_HEADER_INC_DIR "${torc_SOURCE_DIR}/costs/include")

set(COST_HEADER_LIST
    "${COST_HEADER_INC_DIR}/base_cost.h"
    "${COST_HEADER_INC_DIR}/linear_cost.h"
    "${COST_HEADER_INC_DIR}/quadratic_cost.h"
        include/quadratic_cost.h)

add_library(Costs SHARED
        ${COST_HEADER_LIST})

set_target_properties(Costs PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Costs PUBLIC
        $<BUILD_INTERFACE:${torc_SOURCE_DIR}/costs/include>
        $<INSTALL_INTERFACE:costs/include>)

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)