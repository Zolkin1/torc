find_package(Catch2 3 REQUIRED)

message(STATUS "Building tests")

include(Catch)

set(COST_PATH, "../costs/")

#add_library(Costs ${COST_PATH}linear_cost.cpp  ${COST_PATH}linear_cost.h  ${COST_PATH}base_cost.h)
add_executable(CostTests cost_tests.cpp)
target_include_directories(CostTests PRIVATE ../costs/)
message("test" ${COST_PATH})
link_directories(${COST_PATH})
target_link_libraries(CostTests PRIVATE Cost)
