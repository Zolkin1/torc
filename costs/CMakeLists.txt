message(STATUS "Building costs")

set(COST_HEADER_LIST
    "${torc_SOURCE_DIR}/costs/include/base_cost.h"
    "${torc_SOURCE_DIR}/costs/include/linear_cost.h")

add_library(Costs SHARED
        linear_cost.cpp
        ${COST_HEADER_LIST})

target_include_directories(Costs PUBLIC
        $<BUILD_INTERFACE:${torc_SOURCE_DIR}/costs/include>
        $<INSTALL_INTERFACE:costs/include>)

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)