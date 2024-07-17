message(STATUS "Configuring Controllers")

set(CONTROLLERS_INC "${torc_SOURCE_DIR}/controllers/include")
set(CONTROLLERS_HEADER_LIST
        "${CONTROLLERS_INC}/whole_body_qp_controller.h")

set(CONTROLLERS_SOURCES_LIST
    whole_body_qp_controller.cpp
    ${CONTROLLERS_HEADER_LIST})
