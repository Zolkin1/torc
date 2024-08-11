message(STATUS "Configuring sample planner")

set(SAMPLE_INC "${torc_SOURCE_DIR}/sample_planner/include")
set(SAMPLE_HEADER_LIST
    "${SAMPLE_INC}/cross_entropy.h"
    "${SAMPLE_INC}/simulation_dispatcher.h")

add_library(SamplePlanner
    cross_entropy.cpp
    simulation_dispatcher.cpp
    ${SAMPLE_HEADER_LIST})

target_include_directories(SamplePlanner PUBLIC
    ${SAMPLE_INC})

# ----------- MPC ----------- #
# Needed for the trajectory and contact schedule stuff
target_link_libraries(SamplePlanner PUBLIC Mpc)

# ----------- ABSL -----------
#if (NOT TARGET absl::log)
#    message(STATUS "SamplePlanner: `absl` targets not found. Attempting to fetch contents...")
#    FetchContent_Declare(
#            abseil-cpp
#            GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
#            GIT_TAG        origin/master
#    )
#    FetchContent_MakeAvailable(abseil-cpp)
#else()
#    message(STATUS "SamplePlanner: `absl` targets found.")
#endif()

target_link_libraries(SamplePlanner PUBLIC absl::log absl::log_initialize)

# ------- Mujoco ------- #
if (NOT TARGET mujoco-lib)
    include(FetchContent)
    cmake_policy(SET CMP0135 NEW)
    set(MUJOCO_VERSION "3.1.6" CACHE STRING "mujoco version")
    set(COMP_ARCH "x86_64" CACHE STRING "computer architecture")

    FetchContent_Declare(
            mujoco
            URL https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-${COMP_ARCH}.tar.gz
    )

    FetchContent_MakeAvailable(mujoco)

    # ------- Setup the Mujoco library ------- #
    message(STATUS "mujoco source dir: ${mujoco_SOURCE_DIR}")
    find_library(MUJOCO_LIB mujoco REQUIRED HINTS ${mujoco_SOURCE_DIR}/lib)
    add_library(mujoco-lib SHARED IMPORTED GLOBAL)
    target_include_directories(mujoco-lib INTERFACE ${mujoco_SOURCE_DIR}/include)
    target_include_directories(mujoco-lib INTERFACE ${mujoco_SOURCE_DIR}/include/mujoco)

    set_property(TARGET mujoco-lib PROPERTY IMPORTED_LOCATION ${mujoco_SOURCE_DIR}/lib/libmujoco.so)

    target_link_libraries(SamplePlanner PUBLIC mujoco-lib)
endif ()

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
target_link_libraries(SamplePlanner PUBLIC BS_thread_pool)

# TODO: Add in the mujoco dependency