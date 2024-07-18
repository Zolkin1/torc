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
