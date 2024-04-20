cmake_minimum_required(VERSION 3.16)

project(torc
        VERSION 0.1
        LANGUAGES CXX)

message(STATUS "Building TORC Project")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")

if(CMAKE_PROJECT_NAME STREQUAL torc AND BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
endif()

if(CMAKE_PROJECT_NAME STREQUAL torc AND BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

#include_directories(costs)
#include_directories(tests)

add_subdirectory(models)
add_subdirectory(costs)
add_subdirectory(constraints)
add_subdirectory(tests)

add_executable(CostTests costs/cost_tests.cpp)
target_include_directories(CostTests PRIVATE costs)
#add_dependencies(CostTests Cost)

set(COST_HEADER_LIST
        "costs/linear_cost.h"
        "costs/base_cost.h")
add_library(Cost costs/linear_cost.cpp ${COST_HEADER_LIST})

target_link_libraries(CostTests PUBLIC Cost)


# TODO: Figure out the install structure for the library