message(STATUS "Configuring examples")

add_executable(mpc_app_test mpc_app_tests.cpp
        mpc_hpipm_example.cpp)
target_link_libraries(mpc_app_test Mpc)

add_executable(mpc_quad_test mpc_quad_test.cpp)
target_link_libraries(mpc_quad_test Mpc)

add_executable(mpc_g1_test mpc_g1_app_test.cpp)
target_link_libraries(mpc_g1_test Mpc)

add_executable(mpc_hpipm mpc_hpipm_example.cpp)
target_link_libraries(mpc_hpipm Mpc)