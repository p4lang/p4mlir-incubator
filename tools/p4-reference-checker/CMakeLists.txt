cmake_minimum_required(VERSION 3.10)

project(Checker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(p4-reference-checker main.cpp)

target_compile_options(p4-reference-checker PRIVATE -Wall -Wextra -Wpedantic)

install(TARGETS p4-reference-checker DESTINATION bin)
