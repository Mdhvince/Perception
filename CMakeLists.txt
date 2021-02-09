cmake_minimum_required(VERSION 3.5)

project(AerialRobotics VERSION 1.0.0)

add_definitions(-std=c++17)
set(CXX_FLAGS "-Wall")


set(EXECUTABLE app)

add_executable(${EXECUTABLE} main.cpp)