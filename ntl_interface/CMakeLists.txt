cmake_minimum_required(VERSION 3.29)
project(ntl_wrapper)

set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# add native flags for release build:
if(CMAKE_BUILD_TYPE MATCHES "^Rel")
    message(STATUS "Adding native compile options for Release build")
    add_compile_options(-mtune=native -march=native -ffp-contract=fast)
endif()

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
pybind11_add_module(ntl_wrapper main.cpp lll_utils.cpp)
target_link_libraries(ntl_wrapper PRIVATE "ntl" Eigen3::Eigen)

enable_testing()
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(GTest REQUIRED)
add_executable(ntl_wrapper_tests main_tests.cpp)
target_include_directories(ntl_wrapper_tests PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(ntl_wrapper_tests GTest::gtest_main pybind11::embed)

gtest_discover_tests(ntl_wrapper_tests)

add_executable(lll_apx_tests lll_apx_tests.cpp lll_utils.cpp)
target_link_libraries(lll_apx_tests GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(lll_apx_tests)

add_executable(conversion_tests conversion_tests.cpp)
target_include_directories(conversion_tests PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(conversion_tests GTest::gtest_main pybind11::embed)
gtest_discover_tests(conversion_tests)
