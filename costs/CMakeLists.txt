message(STATUS "Building costs")

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)

add_library(Costs ../costs/linear_cost.cpp ../costs/linear_cost.h ../costs/base_cost.h)
