message(STATUS "Building solver interfaces")

set(SOLVERS_INC "${torc_SOURCE_DIR}/solver_interfaces/include")
set(SOLVER_HEADER_LIST
    "${SOLVERS_INC}/ipopt_interface.h"
    "${SOLVERS_INC}/ipopt.h"
    "${SOLVERS_INC}/solver_status.h"
    "${SOLVERS_INC}/clarabel_interface.h")

add_library(SolverInterfaces
    ipopt_interface.cpp
    ipopt.cpp
    solver_status.cpp
    clarabel_interface.cpp)

target_include_directories(SolverInterfaces PUBLIC
    $<BUILD_INTERFACE:${SOLVERS_INC}>
    $<INSTALL_INTERFACE:solver_interfaces/include>)

find_package(Eigen3 REQUIRED)
target_link_libraries(SolverInterfaces PUBLIC Eigen3::Eigen)

#----------- IPOPT ----------- #
set(IPOPT_INC_DIR /usr/local/include/coin-or  CACHE STRING "Directory where IPOPT header file are located.")
set(IPOPT_LIB_DIR /usr/local/lib CACHE STRING "Directory where IPOPT library is installed")

if(EXISTS ${IPOPT_INC_DIR})
    message(STATUS "IPOPT include directory found as ${IPOPT_INC_DIR}.")
else ()
    message(SEND_ERROR "IPOPT include directory not found. Please set IPOPT_INC_DIR after installed IPOPT.")
endif ()

if(EXISTS ${IPOPT_LIB_DIR})
    message(STATUS "IPOPT library directory found as ${IPOPT_LIB_DIR}.")
else ()
    message(SEND_ERROR "IPOPT library directory not found. Please set IPOPT_INC_DIR after installed IPOPT.")
endif ()

target_include_directories(SolverInterfaces PUBLIC
        $<BUILD_INTERFACE:${IPOPT_INC_DIR}>
        $<INSTALL_INTERFACE:${IPOPT_INC_DIR}>)

link_directories(${IPOPT_LIB_DIR})

target_link_libraries(SolverInterfaces PUBLIC
        ${IPOPT_LIB_DIR}/libipopt.so.3
        ${IPOPT_LIB_DIR}/libcoinmumps.so.3)

#----------- Clarabel ----------- #
set(CLARABEL_LIB_PATH "~/Clarabel.cpp/rust_wrapper/target/release/libclarabel_c.so"
        CACHE STRING "Path to the clarabel shared library")
set(CLARABEL_INCLUDE_DIR CACHE STRING "~/Clarabel.cpp/include")

add_library(clarabel STATIC IMPORTED)
set_property(TARGET clarabel PROPERTY
        IMPORTED_LOCATION "${CLARABEL_LIB_PATH}")
target_include_directories(SolverInterfaces PUBLIC "${CLARABEL_INCLUDE_DIR}")
target_link_libraries(SolverInterfaces PUBLIC clarabel)
