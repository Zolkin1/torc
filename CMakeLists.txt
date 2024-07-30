cmake_minimum_required(VERSION 3.14)

project(torc
        VERSION 0.1
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # this ensures -std=c++11 not -std=gnu++11

add_compile_options(-Wall)

message(STATUS "Configuring TORC Project")
message(STATUS "CMake Version: " ${CMAKE_VERSION})

set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(BUILD_TESTING True)

option(BUILD_WITH_IPOPT "Build with IPOPT interface" OFF)
option(BUILD_WITH_OSQP "Build with OSQP interface" OFF)
option(BUILD_WITH_CLARABEL "Build with Clarabel interface" OFF)
option(BUILD_WITH_PINOCCHIO "Build with Pinocchio" ON)

message(STATUS "BUILD_WITH_IPOPT: " ${BUILD_WITH_IPOPT})
message(STATUS "BUILD_WITH_OSQP: " ${BUILD_WITH_OSQP})
message(STATUS "BUILD_WITH_CLARABEL: " ${BUILD_WITH_CLARABEL})
message(STATUS "BUILD_WITH_PINOCCHIO: " ${BUILD_WITH_PINOCCHIO})

if(CMAKE_PROJECT_NAME STREQUAL torc AND BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
endif()

if(CMAKE_PROJECT_NAME STREQUAL torc AND BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

add_subdirectory(models)
add_subdirectory(functions)
add_subdirectory(constraints)
#add_subdirectory(solver_interfaces)
add_subdirectory(mpc)
#add_subdirectory(controllers)
add_subdirectory(utils)


add_executable(mpc_app_test mpc_app_tests.cpp)
target_link_libraries(mpc_app_test Mpc)
# TODO: Figure out the install structure for the library
# TODO: Make different linking targets if they only want part of the library