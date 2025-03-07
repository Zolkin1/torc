message(STATUS "Building tests")

enable_testing()

include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.4 # or a later release
)

FetchContent_MakeAvailable(Catch2)
include(Catch)

# ---------- Model Tests ---------- #
add_executable(rigid_body_tests rigid_body_tests.cpp)
target_link_libraries(rigid_body_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(rigid_body_tests PUBLIC Models)

add_executable(srb_tests srb_tests.cpp)
target_link_libraries(srb_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(srb_tests PUBLIC Models)

catch_discover_tests(rigid_body_tests)
catch_discover_tests(srb_tests)

# ----------- Utils Tests ---------- #
add_executable(utils_tests utils_tests.cpp)
target_link_libraries(utils_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(utils_tests PUBLIC Utils)
catch_discover_tests(utils_tests)

# --------- MPC Tests --------- #
add_executable(mpc_tests mpc_tests.cpp mpc_test_class.cpp mpc_test_class.h
        hpipm_mpc_test.cpp)
target_link_libraries(mpc_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(mpc_tests PUBLIC Mpc)
catch_discover_tests(mpc_tests)

# --------- AD Interface Tests --------- #
add_executable(ad_tests ad_interface_tests.cpp)
target_link_libraries(ad_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(ad_tests PUBLIC Autodiff)
catch_discover_tests(ad_tests)

# --------- HPIPM Tests --------- #
add_executable(mpc_hpipm_tests hpipm_mpc_test.cpp constraints/DynamicsConstraintsTest.cpp constraints/DynamicsConstraintsTest.h)
target_link_libraries(mpc_hpipm_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(mpc_hpipm_tests PUBLIC Mpc)
catch_discover_tests(mpc_hpipm_tests)

# Dynamics
add_executable(dynamics_tests dynamics_tests.cpp)
target_link_libraries(dynamics_tests PUBLIC Models SamplePlanner Mpc) # Sample planner gives access to Mujoco

# Copy additional test data to the build directory
file(COPY test_data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# --------- LPF Tests --------- #
add_executable(lpf_tests LowPassFilterTests.cpp)
target_link_libraries(lpf_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(lpf_tests PUBLIC StateEstimation)
catch_discover_tests(lpf_tests)

# --------- Frame Transform Tests --------- #
add_executable(frame_tests frame_transform_tests.cpp)
target_link_libraries(frame_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(frame_tests PUBLIC Models)
catch_discover_tests(frame_tests)
