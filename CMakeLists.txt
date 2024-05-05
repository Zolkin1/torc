cmake_minimum_required(VERSION 3.16)

project(torc
        VERSION 0.1
        LANGUAGES CXX)

message(STATUS "Building TORC Project")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(BUILD_TESTING True)

if(CMAKE_PROJECT_NAME STREQUAL torc AND BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
endif()

if(CMAKE_PROJECT_NAME STREQUAL torc AND BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

add_subdirectory(models)
add_subdirectory(costs)
add_subdirectory(constraints)

# TODO: Figure out the install structure for the library