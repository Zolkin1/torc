find_package(Catch2 3 REQUIRED)

message(STATUS "Building tests")

include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.4 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(cost_tests cost_tests.cpp)
target_link_libraries(cost_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(cost_tests PUBLIC Costs)

include(Catch)
catch_discover_tests(cost_tests)