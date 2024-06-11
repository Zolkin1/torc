message(STATUS "Building Utils")

set(UTILS_INC "${torc_SOURCE_DIR}/utils/include")
set(UTILS_HEADER_LIST
    "${UTILS_INC}/torc_timer.h")

add_library(Utils
    torc_timer.cpp)

target_include_directories(Utils PUBLIC
        $<BUILD_INTERFACE:${UTILS_INC}>
        $<INSTALL_INTERFACE:utils/include>)