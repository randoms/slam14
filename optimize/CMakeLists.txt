project(optimize)
cmake_minimum_required(VERSION 3.16)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    main.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
)