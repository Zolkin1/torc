message(STATUS "Building MPC")

set(MPC_INC "${torc_SOURCE_DIR}/mpc/include")
set(MPC_HEADER_LIST
    "${MPC_INC}/mpc_base.h"
    "${MPC_INC}/trajectory.h"
    "${MPC_INC}/mpc_contact.h")

set(MPC_SOURCE_LIST
        mpc_base.cpp
        mpc_contact.cpp
        trajectory.cpp)

add_library(Mpc
    ${MPC_SOURCE_LIST}
    ${MPC_HEADER_LIST})

set_target_properties(Mpc PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Mpc PUBLIC
        $<BUILD_INTERFACE:${MPC_INC}>
        $<INSTALL_INTERFACE:mpc/include>)

# ------- Model Dependency ------ #
target_link_libraries(Mpc PUBLIC Models)


