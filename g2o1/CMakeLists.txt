project(g2o_test)
cmake_minimum_required(VERSION 3.16)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(G2O REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${G2O_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    main.cpp
)

SET (G2O_LIBS g2o_csparse_extension g2o_cli g2o_core g2o_interface g2o_parser g2o_solver_cholmod
g2o_solver_csparse g2o_solver_dense g2o_solver_pcg g2o_stuff g2o_types_icp
g2o_types_sba g2o_types_sim3 g2o_types_slam2d g2o_types_slam3d)

target_link_libraries(${PROJECT_NAME}
    ${G2O_LIBS}
    ${OpenCV_LIBRARIES}
)