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

# ---------- Function Tests ------- #
add_executable(function_tests
                function_tests.cpp
                test_fn.h)
target_link_libraries(function_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(function_tests PUBLIC Functions)

# ---------- Constraint Tests ---------- #
add_executable(constraint_tests constraint_tests.cpp)
target_link_libraries(constraint_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(constraint_tests PUBLIC Constraints)

# ---------- Model Tests ---------- #
if (BUILD_WITH_PINOCCHIO)
    add_executable(rigid_body_tests rigid_body_tests.cpp)
    target_link_libraries(rigid_body_tests PRIVATE Catch2::Catch2WithMain)
    target_link_libraries(rigid_body_tests PUBLIC Models)

    add_executable(srb_tests srb_tests.cpp)
    target_link_libraries(srb_tests PRIVATE Catch2::Catch2WithMain)
    target_link_libraries(srb_tests PUBLIC Models)

    catch_discover_tests(rigid_body_tests)
    catch_discover_tests(srb_tests)
endif ()

# ---------- IPOPT Tests ---------- #
if (BUILD_WITH_IPOPT)
    add_executable(ipopt_tests ipopt_tests.cpp)
    target_link_libraries(ipopt_tests PRIVATE Catch2::Catch2WithMain)
    target_link_libraries(ipopt_tests PUBLIC SolverInterfaces)
    catch_discover_tests(ipopt_tests)
endif ()

# ---------- OSQPInterface Tests ---------- #
if (BUILD_WITH_OSQP)
    add_executable(osqp_tests osqp_tests.cpp)
    target_link_libraries(osqp_tests PRIVATE Catch2::Catch2WithMain)
    target_link_libraries(osqp_tests PUBLIC SolverInterfaces)
    catch_discover_tests(osqp_tests)
endif ()
# ----------- Utils Tests ---------- #
add_executable(utils_tests utils_tests.cpp)
target_link_libraries(utils_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(utils_tests PUBLIC Utils)
catch_discover_tests(utils_tests)

# --------- Clarabel Tests --------- #
if (BUILD_WITH_CLARABEL)
    add_executable(clarabel_tests clarabel_tests.cpp)
    target_link_libraries(clarabel_tests PRIVATE Catch2::Catch2WithMain)
    target_link_libraries(clarabel_tests PUBLIC SolverInterfaces)
    catch_discover_tests(clarabel_tests)
endif ()

## --------- Controller Tests --------- #
#add_executable(controller_tests controller_tests.cpp)
#target_link_libraries(controller_tests PRIVATE Catch2::Catch2WithMain)
#target_link_libraries(controller_tests PUBLIC Controllers)

# --------- MPC Tests --------- #
add_executable(mpc_tests mpc_tests.cpp mpc_test_class.cpp)
target_link_libraries(mpc_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(mpc_tests PUBLIC Mpc)
catch_discover_tests(mpc_tests)

# Copy additional test data to the build directory
file(COPY test_data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


include(Catch)
catch_discover_tests(function_tests)