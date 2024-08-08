message(STATUS "Configuring sample planner")

set(SAMPLE_INC "${torc_SOURCE_DIR}/sample_planner/include")
set(SAMPLE_HEADER_LIST
    "${SAMPLE_INC}/cross_entropy.h")

add_library(SamplePlanner
    cross_entropy.cpp
    ${SAMPLE_HEADER_LIST})

target_include_directories(SamplePlanner PUBLIC
    ${SAMPLE_INC})

# ----------- Thread Pool ----------- #
# For now use with CPM
file(
        DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
        EXPECTED_HASH SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
CPMAddPackage(
        NAME BS_thread_pool
        GITHUB_REPOSITORY bshoshany/thread-pool
        VERSION 4.1.0)
add_library(BS_thread_pool INTERFACE)
target_include_directories(BS_thread_pool INTERFACE ${BS_thread_pool_SOURCE_DIR}/include)
target_link_libraries(SamplePlanner BS_thread_pool)

# TODO: Add in the mujoco dependency