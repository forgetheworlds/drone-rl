cmake_minimum_required(VERSION 3.16)
project(drone_navigation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -ffp-contract=off)

add_executable(benchmark src/benchmark.cpp)
target_include_directories(benchmark PRIVATE include)

add_executable(runtime_test tests/runtime_test.cpp)
target_include_directories(runtime_test PRIVATE include)

enable_testing()
add_test(NAME runtime COMMAND runtime_test ${CMAKE_SOURCE_DIR}/assets/raptor.bin)
add_test(NAME mission
         COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/run_missions.py public
                 --only public-visible-Simple-8 --binary $<TARGET_FILE:benchmark>
                 --output ${CMAKE_BINARY_DIR}/test-results --require-success)
