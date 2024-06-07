message(STATUS "Building tests")

include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.4 # or a later release
)

FetchContent_MakeAvailable(Catch2)

# ---------- Cost Tests ---------- #
add_executable(cost_tests
                function_tests.cpp
                test_fn.h)
target_link_libraries(cost_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(cost_tests PUBLIC Functions)

# ---------- Model Tests ---------- #
add_executable(rigid_body_tests rigid_body_tests.cpp)
target_link_libraries(rigid_body_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(rigid_body_tests PUBLIC Models)

add_executable(srb_tests srb_tests.cpp)
target_link_libraries(srb_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(srb_tests PUBLIC Models)

# Copy additional test data to the build directory
file(COPY test_data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

include(Catch)
catch_discover_tests(cost_tests)