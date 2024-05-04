message(STATUS "Building models")

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)
#target_link_libraries(models PUBLIC Eigen3::Eigen)

#----------- Pinocchio -----------#
find_package(pinocchio)
#target_link_libraries(models PRIVATE pinocchio)

set(MODELS_INC "${torc_SOURCE_DIR}/models/include")
set(MODELS_HEADER_LIST "${MODELS_INC}/model.h")

add_library(Models SHARED
        ${MODELS_HEADER_LIST})

set_target_properties(Models PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(Models PUBLIC
        $<BUILD_INTERFACE:${MODELS_INC}>
        $<INSTALL_INTERFACE:models/include>)