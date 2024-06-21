message(STATUS "Building Utils")

set(UTILS_INC "${torc_SOURCE_DIR}/utils/include")
set(UTILS_HEADER_LIST
    "${UTILS_INC}/torc_timer.h"
    "${UTILS_INC}/eigen_utils.h"
    "${UTILS_INC}/sparse_matrix_builder.h")

add_library(Utils
    torc_timer.cpp
    sparse_matrix_builder.cpp)

set_target_properties(Utils PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Utils PUBLIC
        $<BUILD_INTERFACE:${UTILS_INC}>
        $<INSTALL_INTERFACE:utils/include>)

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)