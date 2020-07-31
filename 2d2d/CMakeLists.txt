project(2d2d)
cmake_minimum_required(VERSION 3.16)

find_package(OpenCV 3 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    main.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
)