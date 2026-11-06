cmake_minimum_required(VERSION 3.20)
project(drone_rl LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# This checkout is the drone-rl root: grade/reference include navigation/ from here.
set(DRONE_RL_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "Path to this drone-rl tree")

find_package(Threads REQUIRED)
find_package(BLAS)
# Fall back to the system libblas when CMake's FindBLAS has no development package.
if(NOT BLAS_FOUND AND NOT APPLE)
  find_library(BLAS_LIBRARIES NAMES openblas blas openblas.so.0 blas.so.3
    PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib64
    PATH_SUFFIXES openblas-pthread)
  # Runtime-only package may ship libblas.so.3 without the unsuffixed .so symlink.
  if(NOT BLAS_LIBRARIES AND EXISTS /usr/lib/x86_64-linux-gnu/libblas.so.3)
    set(BLAS_LIBRARIES /usr/lib/x86_64-linux-gnu/libblas.so.3)
  endif()
  if(BLAS_LIBRARIES)
    set(BLAS_FOUND TRUE)
    message(STATUS "Using BLAS library: ${BLAS_LIBRARIES}")
  endif()
endif()

# RAPTOR + VisibleWorld runtime smoke (portable C++17).
add_executable(runtime_test tests/runtime_test.cpp)
target_include_directories(runtime_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
enable_testing()
add_test(NAME runtime COMMAND runtime_test ${CMAKE_SOURCE_DIR}/assets/raptor.bin ${CMAKE_SOURCE_DIR}/missions_smoke/public/Simple.3dmap)

if(BLAS_FOUND OR APPLE)
  add_executable(rlv2_grade rlv2/grade.cpp)
  target_include_directories(rlv2_grade PRIVATE
    ${CMAKE_SOURCE_DIR}/rlv2
    ${CMAKE_SOURCE_DIR}/include)
  target_compile_options(rlv2_grade PRIVATE -O2 -Wall -Wextra)
  target_link_libraries(rlv2_grade PRIVATE Threads::Threads ${BLAS_LIBRARIES})
  if(APPLE)
    target_link_libraries(rlv2_grade PRIVATE "-framework Accelerate")
    target_compile_definitions(rlv2_grade PRIVATE ACCELERATE_NEW_LAPACK)
  endif()

  add_executable(rlv2_train_cpu rlv2/train_cpu.cpp)
  target_include_directories(rlv2_train_cpu PRIVATE
    ${CMAKE_SOURCE_DIR}/rlv2
    ${CMAKE_SOURCE_DIR}/rlv2/cpu
    ${CMAKE_SOURCE_DIR})
  target_compile_options(rlv2_train_cpu PRIVATE -O2 -Wall -Wno-attributes -Wno-unused-variable)
  target_link_libraries(rlv2_train_cpu PRIVATE Threads::Threads ${BLAS_LIBRARIES})
  if(APPLE)
    target_link_libraries(rlv2_train_cpu PRIVATE "-framework Accelerate")
    target_compile_definitions(rlv2_train_cpu PRIVATE ACCELERATE_NEW_LAPACK)
  endif()

  add_executable(rlv2_reference rlv2/reference.cpp)
  target_include_directories(rlv2_reference PRIVATE
    ${CMAKE_SOURCE_DIR}/rlv2
    ${CMAKE_SOURCE_DIR}/include)
  target_compile_options(rlv2_reference PRIVATE -O2 -Wall -Wextra)
endif()

# Metal high-throughput sim (macOS only). See docs/MAC_METAL.md.
if(APPLE)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  add_executable(rlv2_metal rlv2/rlv2.mm)
  target_compile_options(rlv2_metal PRIVATE -O3 -fobjc-arc -Wall -Wextra)
  target_compile_definitions(rlv2_metal PRIVATE SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  target_include_directories(rlv2_metal PRIVATE ${CMAKE_SOURCE_DIR}/rlv2 ${CMAKE_SOURCE_DIR})
  target_link_libraries(rlv2_metal PRIVATE "-framework Foundation" "-framework Metal")

  add_executable(rlv2_train_metal rlv2/train.mm)
  target_compile_options(rlv2_train_metal PRIVATE -O3 -fobjc-arc -Wall -Wextra -Wno-unused-parameter)
  target_compile_definitions(rlv2_train_metal PRIVATE SOURCE_DIR="${CMAKE_SOURCE_DIR}" ACCELERATE_NEW_LAPACK)
  target_include_directories(rlv2_train_metal PRIVATE ${CMAKE_SOURCE_DIR}/rlv2 ${CMAKE_SOURCE_DIR})
  target_link_libraries(rlv2_train_metal PRIVATE
    "-framework Foundation" "-framework Metal" "-framework Accelerate")
endif()
