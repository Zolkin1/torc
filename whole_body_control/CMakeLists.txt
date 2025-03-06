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

# ----------- ProxQp ----------- #
find_package(proxsuite REQUIRED)
target_link_libraries(Wbc PUBLIC proxsuite::proxsuite)

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
