message(STATUS "Configuring MPC")

set(MPC_INC "${torc_SOURCE_DIR}/mpc/include")
set(MPC_HEADER_LIST
#    "${MPC_INC}/mpc_base.h"
    "${MPC_INC}/trajectory.h"
    "${MPC_INC}/full_order_mpc.h")
#    "${MPC_INC}/mpc_contact.h")

set(MPC_SOURCE_LIST
#        mpc_base.cpp
        full_order_mpc.cpp
#        mpc_contact.cpp
        trajectory.cpp)

add_library(Mpc
    ${MPC_SOURCE_LIST}
    ${MPC_HEADER_LIST})

set_target_properties(Mpc PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Mpc PUBLIC
        $<BUILD_INTERFACE:${MPC_INC}>
        $<INSTALL_INTERFACE:mpc/include>)

# ------- Model ------ #
target_link_libraries(Mpc PUBLIC Models)

# ------- Constraints ------ #
target_link_libraries(Mpc PUBLIC Constraints)

# ------- Solver ------ #
#target_link_libraries(Mpc PUBLIC SolverInterfaces)
if (BUILD_WITH_OSQP)
    find_package(osqp)
    if (NOT TARGET osqpstatic)  # This is what osqp-cpp looks for
        message(STATUS "osqpstatic target not found, checking osqp::osqpstatic...")
        if (NOT TARGET osqp::osqpstatic)
            message(STATUS "Could not find osqp::osqpstatic, osqp-cpp will try to fetch it...")
        else ()
            message(STATUS "Found osqp::osqpstatic. Using it in osqp-cpp.")
            add_library(osqpstatic ALIAS osqp::osqpstatic)
        endif ()
    endif ()
endif ()

include(FetchContent)

set(OSQP-CPP_BUILD_TESTS OFF CACHE BOOL "Build OSQP-CPP without tests")
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Variable for OSQP-CPP ABSL dep")

FetchContent_Declare(
        osqp-cpp
        GIT_REPOSITORY https://github.com/google/osqp-cpp.git
        GIT_TAG 43433736334d6b515ea4b0247156fea9e56c0d3f
)

FetchContent_MakeAvailable(osqp-cpp)
target_link_libraries(Mpc PUBLIC osqp-cpp)

# ------- Utils ------ #
target_link_libraries(Mpc PUBLIC Utils)

#----------- YAML ----------- #
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0 # note all other versions are of the form yaml-cpp-x.y.z
)
FetchContent_GetProperties(yaml-cpp)

if(NOT yaml-cpp_POPULATED)
    message(STATUS "Fetching yaml-cpp...")
    FetchContent_Populate(yaml-cpp)
    add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

target_link_libraries(Mpc PRIVATE yaml-cpp::yaml-cpp)

