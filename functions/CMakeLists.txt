message(STATUS "Building costs")

set(FN_HEADER_INC_DIR "${torc_SOURCE_DIR}/functions/include")

set(FN_HEADER_LIST
    "${FN_HEADER_INC_DIR}/explicit_fn.h"
    "${FN_HEADER_INC_DIR}/linear_fn.h"
    "${FN_HEADER_INC_DIR}/quadratic_fn.h"
    "${FN_HEADER_INC_DIR}/finite_diff_fn.h"
    "${FN_HEADER_INC_DIR}/autodiff_fn.h")
add_library(Functions SHARED ${FN_HEADER_LIST})

set_target_properties(Functions PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Functions PUBLIC
        $<BUILD_INTERFACE:${FN_HEADER_INC_DIR}>
        $<INSTALL_INTERFACE:functions/include>)

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)
target_link_libraries(Functions PUBLIC Eigen3::Eigen)

# pinocchio
list(APPEND CMAKE_PREFIX_PATH /opt/openrobots/)
find_package(pinocchio 3.0.0 REQUIRED)
target_link_libraries(Functions PUBLIC pinocchio::pinocchio)