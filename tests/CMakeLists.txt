message(STATUS "Building tests")

include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.4 # or a later release
)

FetchContent_MakeAvailable(Catch2)

# ---------- Cost Tests ---------- #
add_executable(cost_tests cost_tests.cpp)
target_link_libraries(cost_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(cost_tests PUBLIC Costs)

# ---------- Model Tests ---------- #
add_executable(model_tests model_tests.cpp)
target_link_libraries(model_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(model_tests PUBLIC Models)


# Copy additional test data to the build directory
file(COPY test_data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

include(Catch)
catch_discover_tests(cost_tests)