cmake_minimum_required(VERSION 3.5)

project(AerialRobotics VERSION 1.0.0)

add_definitions(-std=c++17)
set(CXX_FLAGS "-Wall")

find_package(OpenCV REQUIRED)

set(EXECUTABLE geom)
add_executable(${EXECUTABLE} main.cpp)

target_link_libraries(${EXECUTABLE} PRIVATE ${OpenCV_LIBS})

