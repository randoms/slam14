project(sophus_test)
cmake_minimum_required(VERSION 3.16)

find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)


include_directories(
    ${SOPHUS_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    main.cpp
)