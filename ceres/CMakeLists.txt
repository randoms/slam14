project(ceres_test)
cmake_minimum_required(VERSION 3.16)

find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CERES_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    main.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)