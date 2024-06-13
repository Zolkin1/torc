message(STATUS "Building constraints")

set(CONSTRAINT_INC_DIR "${torc_SOURCE_DIR}/constraints/include")

set(CONSTRAINT_HEADER_LIST
        "${CONSTRAINT_INC_DIR}/constraint.h")
add_library(Constraints SHARED ${CONSTRAINT_HEADER_LIST})
target_link_libraries(Constraints SHARED Functions)

set_target_properties(Constraints PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Constraints PUBLIC
        $<BUILD_INTERFACE:${CONSTRAINT_INC_DIR}>
        $<INSTALL_INTERFACE:constraints/include>)

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)