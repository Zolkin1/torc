message(STATUS "Configuring State Estimation")

set(STATE_EST_INC "${torc_SOURCE_DIR}/state_estimation/include")
set(STATE_EST_HEADER_LIST
        "${STATE_EST_INC}/low_pass_filter.h")

set(STATE_EST_SOURCE_LIST
        low_pass_filter.cpp
)

add_library(StateEstimation
        ${STATE_EST_SOURCE_LIST}
        ${STATE_EST_HEADER_LIST})

set_target_properties(StateEstimation PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(StateEstimation PUBLIC
        $<BUILD_INTERFACE:${STATE_EST_INC}>
        $<INSTALL_INTERFACE:state_estimation/include>)

## ------- Model ------ #
#target_link_libraries(Wbc PUBLIC Models)
#
## ------- CppADInterface ------ #
#target_link_libraries(Wbc PUBLIC Autodiff)

# ------- Eigen ------ #
find_package(Eigen3 REQUIRED)

# ------- Utils ------ #
target_link_libraries(StateEstimation PUBLIC Utils)
