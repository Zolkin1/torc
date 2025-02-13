message(STATUS "Configuring examples")

add_executable(mpc_app_test mpc_app_tests.cpp)
target_link_libraries(mpc_app_test Mpc)

add_executable(mpc_quad_test mpc_quad_test.cpp)
target_link_libraries(mpc_quad_test Mpc)

add_executable(mpc_g1_test mpc_g1_app_test.cpp)
target_link_libraries(mpc_g1_test Mpc)

add_executable(mpc_hpipm mpc_hpipm_example.cpp)
target_link_libraries(mpc_hpipm Mpc)

add_executable(wbc_example wbc_example.cpp)
target_link_libraries(wbc_example Wbc Mpc)

#add_executable(casadi_test casadi_test.cpp)
#target_link_libraries(casadi_test Mpc)

# Mujoco example
# Setup the Mujoco library
set(MUJOCO_VERSION "mujoco-3.1.6" CACHE STRING "mujoco version")
set(MUJOCO_DIR "/home/$ENV{USER}/mujoco/${MUJOCO_VERSION}" CACHE PATH "Path to Mujoco dir")
message(STATUS "MUJOCO_DIR " ${MUJOCO_DIR})

find_library(MUJOCO_LIB mujoco REQUIRED HINTS ${MUJOCO_DIR}/lib)
message(STATUS "MUJOCO_LIB: ${MUJOCO_LIB}")

add_library(mujoco SHARED IMPORTED GLOBAL)
set_target_properties(mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_DIR}/include;${MUJOCO_DIR}/include/mujoco")
#target_include_directories(mujoco INTERFACE ${MUJOCO_DIR}/include)
#target_include_directories(mujoco INTERFACE ${MUJOCO_DIR}/include/mujoco)

#find_package(mujoco REQUIRED PATHS ${MUJOCO_DIR} NO_DEFAULT_PATH)

find_package(glfw3 REQUIRED)

add_executable(mpc_mujoco mpc_mujoco.cpp)
target_link_libraries(mpc_mujoco PUBLIC Mpc mujoco glfw StepPlanning)

# Parallel Example
add_executable(mpc_parallel parallel_mpc_example.cpp)
target_link_libraries(mpc_parallel PUBLIC Mpc)

find_package(OpenMP REQUIRED)
target_link_libraries(mpc_parallel PUBLIC OpenMP::OpenMP_CXX)

