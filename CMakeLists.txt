cmake_minimum_required(VERSION 3.14)

project(torc
        VERSION 0.1
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # this ensures -std=c++11 not -std=gnu++11

message(STATUS "Configuring TORC Project")
message(STATUS "CMake Version: " ${CMAKE_VERSION})

#set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(BUILD_TESTING True)

if(CMAKE_PROJECT_NAME STREQUAL torc AND BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests)
endif()

if(CMAKE_PROJECT_NAME STREQUAL torc)
    add_subdirectory(examples)
endif()

add_subdirectory(models)
add_subdirectory(auto_diff)
add_subdirectory(mpc)
add_subdirectory(utils)
#add_subdirectory(whole_body_control)   # TODO: Remove OSQP from here to remove the terrible absl dep

# TODO: Figure out the install structure for the library
# TODO: Make different linking targets if they only want part of the library
# TODO: Make all the targets torc:: namespaced