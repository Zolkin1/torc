message(STATUS "Building solver interfaces")

set(SOLVERS_INC "${torc_SOURCE_DIR}/solver_interfaces/include")
set(SOLVER_HEADER_LIST
    "${SOLVERS_INC}/solver_status.h")



set(SOLVER_SOURCES_LIST
    solver_status.cpp)

#----------- IPOPT ----------- #
if (BUILD_WITH_IPOPT)
    message(STATUS "Building with IPOPT")

    list(APPEND SOLVER_SOURCES_LIST
            ipopt_interface.cpp
            ipopt.cpp)

    list(APPEND SOLVER_HEADER_LIST
            "${SOLVERS_INC}/ipopt_interface.h"
            "${SOLVERS_INC}/ipopt.h")
endif ()

# ------------ OSQPInterface & OSQPInterface Eigen ---------- #
if (BUILD_WITH_OSQP)
    message(STATUS "Building with OSQP")

    list(APPEND SOLVER_SOURCES_LIST
            osqp_interface.cpp)

    list(APPEND SOLVER_HEADER_LIST
            "${SOLVERS_INC}/osqp_interface.h")
endif ()

#----------- Clarabel ----------- #
if (BUILD_WITH_CLARABEL)
    message(STATUS "Building with Clarabel")
    list(APPEND SOLVER_SOURCES_LIST
            clarabel_interface.cpp)

    list(APPEND SOLVER_HEADER_LIST
            "${SOLVERS_INC}/clarabel_interface.h")
endif ()

# -------- Create the library -------- #
add_library(SolverInterfaces
        ${SOLVER_SOURCES_LIST}
        ${SOLVER_HEADER_LIST})

target_include_directories(SolverInterfaces PUBLIC
        $<BUILD_INTERFACE:${SOLVERS_INC}>
        $<INSTALL_INTERFACE:solver_interfaces/include>)

# -------- Eigen -------- #
find_package(Eigen3 REQUIRED)
target_link_libraries(SolverInterfaces PUBLIC Eigen3::Eigen)

# -------- Constraints -------- #
target_link_libraries(SolverInterfaces PUBLIC Constraints)

# -------- Link the necessary files from the dependencies -------- #
if(BUILD_WITH_IPOPT)
    set(IPOPT_INC_PATH /usr/local/include/coin-or  CACHE STRING "Directory where IPOPT header file are located.")
    set(IPOPT_LIB_PATH /usr/local/lib CACHE STRING "Directory where IPOPT library is installed")

    if(EXISTS ${IPOPT_INC_PATH})
        message(STATUS "IPOPT include directory found as ${IPOPT_INC_PATH}.")
    else ()
        message(SEND_ERROR "IPOPT include directory not found. Please set IPOPT_INC_DIR after installed IPOPT.")
    endif ()

    if(EXISTS ${IPOPT_LIB_PATH})
        message(STATUS "IPOPT library directory found as ${IPOPT_LIB_PATH}.")
    else ()
        message(SEND_ERROR "IPOPT library directory not found. Please set IPOPT_INC_DIR after installed IPOPT.")
    endif ()

    target_include_directories(SolverInterfaces PUBLIC
            $<BUILD_INTERFACE:${IPOPT_INC_PATH}>
            $<INSTALL_INTERFACE:${IPOPT_INC_PATH}>)

    link_directories(${IPOPT_LIB_PATH})

    target_link_libraries(SolverInterfaces PUBLIC
            ${IPOPT_LIB_PATH}/libipopt.so.3
            ${IPOPT_LIB_PATH}/libcoinmumps.so.3)
endif()

if (BUILD_WITH_OSQP)
    find_package(osqp REQUIRED)
    find_package(OsqpEigen REQUIRED)
    target_link_libraries(SolverInterfaces PUBLIC osqp::osqp OsqpEigen::OsqpEigen)
endif ()

if (BUILD_WITH_CLARABEL)
    set(CLARABEL_DIR "$ENV{HOME}/Clarabel.cpp" CACHE STRING "Root directory of Clarabel.cpp")
    set(CLARABEL_LIB_PATH "${CLARABEL_DIR}/rust_wrapper/target/release/libclarabel_c.so" CACHE STRING "Path to the clarabel shared library")
    set(CLARABEL_INC_PATH "${CLARABEL_DIR}/include/" CACHE STRING "Path to Clarabel header files")

    add_library(clarabel STATIC IMPORTED)
    set_property(TARGET clarabel PROPERTY IMPORTED_LOCATION "${CLARABEL_LIB_PATH}")
    target_include_directories(SolverInterfaces PUBLIC "${CLARABEL_INC_PATH}")
    target_link_libraries(SolverInterfaces PUBLIC clarabel)
endif ()