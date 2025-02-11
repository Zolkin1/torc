message(STATUS "Configuring WBC")

set(WBC_INC "${torc_SOURCE_DIR}/whole_body_control/include")
set(WBC_HEADER_LIST
        "${WBC_INC}/wbc_controller.h")

set(WBC_SOURCE_LIST
        wbc_controller.cpp
    )

add_library(Wbc
        ${WBC_SOURCE_LIST}
        ${WBC_HEADER_LIST})

set_target_properties(Wbc PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Wbc PUBLIC
        $<BUILD_INTERFACE:${WBC_INC}>
        $<INSTALL_INTERFACE:whole_body_control/include>)

# ------- Model ------ #
target_link_libraries(Wbc PUBLIC Models)

# ------- CppADInterface ------ #
target_link_libraries(Wbc PUBLIC Autodiff)

# ------- Eigen ------ #
find_package(Eigen3 REQUIRED)

# ------- OSQP ------ #
include(FetchContent)

set(OSQP-CPP_BUILD_TESTS OFF CACHE BOOL "Build OSQP-CPP without tests")
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "Variable for OSQP-CPP ABSL dep")

FetchContent_Declare(
        osqp-cpp
        GIT_REPOSITORY https://github.com/google/osqp-cpp.git
        GIT_TAG 43433736334d6b515ea4b0247156fea9e56c0d3f
)

FetchContent_MakeAvailable(osqp-cpp)
target_link_libraries(Wbc PUBLIC osqp-cpp)

#----------- YAML ----------- #
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0 # note all other versions are of the form yaml-cpp-x.y.z
)
FetchContent_MakeAvailable(yaml-cpp)

target_link_libraries(Wbc PUBLIC yaml-cpp::yaml-cpp)

# ------- Utils ------ #
target_link_libraries(Wbc PUBLIC Utils)
