find_package(Catch2 3 REQUIRED)

message(STATUS "Building tests")

include(Catch)

include_directories("../costs/")
#add_library(Costs ../costs/linear_cost.cpp ../costs/linear_cost.h ../costs/base_cost.h)
add_executable(CostTests cost_tests.cpp)
add_dependencies(CostTests Costs)
#target_link_libraries(CostTests PUBLIC Costs)