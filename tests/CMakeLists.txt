find_package(Catch2 3 REQUIRED)

message(STATUS "Building tests")


add_executable(cost_tests cost_tests.cpp)
target_link_libraries(cost_tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(cost_tests PUBLIC Costs)


include(Catch)
catch_discover_tests(cost_tests)