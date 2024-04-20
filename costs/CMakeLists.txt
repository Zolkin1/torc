message(STATUS "Building costs")

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)

add_library(Cost linear_cost.cpp linear_cost.h base_cost.h)