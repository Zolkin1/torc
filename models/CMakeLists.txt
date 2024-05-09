message(STATUS "Building models")

#----------- Eigen ----------- #
find_package(Eigen3 REQUIRED)
#target_link_libraries(models PUBLIC Eigen3::Eigen)

#----------- Pinocchio -----------#
#find_package(pinocchio)
#target_link_libraries(models PRIVATE pinocchio)

