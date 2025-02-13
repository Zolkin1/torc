message(STATUS "Configuring Step Planning")

set(STEP_PLANNING_INC "${torc_SOURCE_DIR}/step_planning/include")
set(STEP_PLANNING_HEADER_LIST
        "${STEP_PLANNING_INC}/step_planner.h")

set(STEP_PLANNING_SOURCE_LIST
        step_planner.cpp
)

add_library(StepPlanning
        ${STEP_PLANNING_SOURCE_LIST}
        ${STEP_PLANNING_HEADER_LIST})

set_target_properties(StepPlanning PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(StepPlanning PUBLIC
        $<BUILD_INTERFACE:${STEP_PLANNING_INC}>
        $<INSTALL_INTERFACE:step_planning/include>)

# ------- Model ------ #
target_link_libraries(StepPlanning PUBLIC Models)

# ------- MPC ------ #
target_link_libraries(StepPlanning PUBLIC Mpc)

# ------- Eigen ------ #
find_package(Eigen3 REQUIRED)

# ------- Utils ------ #
target_link_libraries(StepPlanning PUBLIC Utils)

# ----------- ProxQp ----------- #
find_package(proxsuite REQUIRED)
target_link_libraries(StepPlanning PUBLIC proxsuite::proxsuite)