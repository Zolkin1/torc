message(STATUS "Configuring Controllers")

set(CONTROLLERS_INC "${torc_SOURCE_DIR}/controllers/include")
set(CONTROLLERS_HEADER_LIST
        "${CONTROLLERS_INC}/whole_body_qp_controller.h")

add_library(Controllers
    whole_body_qp_controller.cpp
    ${CONTROLLERS_HEADER_LIST})

target_include_directories(Controllers PUBLIC
        $<BUILD_INTERFACE:${CONTROLLERS_INC}>
        $<INSTALL_INTERFACE:controllers/include>)

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)
target_link_libraries(Controllers PUBLIC Eigen3::Eigen)

#----------- Models ----------- #
target_link_libraries(Controllers PUBLIC Models)

#----------- Solvers ----------- #
target_link_libraries(Controllers PUBLIC SolverInterfaces)

#----------- YAML ----------- #
include(FetchContent)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0 # note all other versions are of the form yaml-cpp-x.y.z
)
FetchContent_GetProperties(yaml-cpp)

if(NOT yaml-cpp_POPULATED)
    message(STATUS "Fetching yaml-cpp...")
    FetchContent_Populate(yaml-cpp)
    add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()

target_link_libraries(Controllers PRIVATE yaml-cpp::yaml-cpp)
