cmake_minimum_required(VERSION 3.26)
project(HPCNas C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_INSTALL_PREFIX $ENV{CONDA_PREFIX})
message(STATUS "CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}")
include_directories(${CMAKE_INSTALL_PREFIX}/include)
add_executable(HPCNas
        src/main.c)
