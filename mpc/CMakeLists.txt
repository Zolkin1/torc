message(STATUS "Building MPC")

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

# ------- Solver ------ #
target_link_libraries(Mpc PUBLIC SolverInterfaces)

# ------- Utils ------ #
target_link_libraries(Mpc PUBLIC Utils)

#----------- YAML ----------- #
include(FetchContent)
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

