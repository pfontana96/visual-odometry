cmake_minimum_required(VERSION 3.0.2)
project(ottobot_pkg)

## Compile as C++14
add_compile_options(-std=c++17)

# find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)

# Find PCL library
# find_package(PCL REQUIRED COMPONENTS common io)
# link_directories(${PCL_LIBRARY_DIRS})
# add_definitions(${PCL_DEFINITIONS})

# Find OpenCV
find_package(OpenCV 4 REQUIRED)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
  message(STATUS "Found OpenMP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OpenMP_FOUND)

# Enable CUDA
find_package(CUDA)
if(CUDA_FOUND)
  enable_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "-arch=sm_53")
    set(CUDA_VERBOSE_BUILD on )
  endif(CMAKE_CUDA_COMPILER)

else()
  message(WARNING "Could not find CUDA")

endif()

# Ottobot Library
file(GLOB_RECURSE OTTO_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

if(CMAKE_CUDA_COMPILER)
  file(GLOB_RECURSE OTTO_GPU_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/src/*.cu")
  set(OTTO_SRC_FILES ${OTTO_SRC_FILES} ${OTTO_GPU_SRC_FILES})
endif(CMAKE_CUDA_COMPILER)

set(OTTO_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/)

add_library(otto ${OTTO_SRC_FILES})

target_include_directories( otto PUBLIC 
  ${OTTO_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${manif_INCLUDE_DIRS}
)

if(CMAKE_CUDA_COMPILER)
  set_target_properties(otto PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(otto PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif(CMAKE_CUDA_COMPILER)

target_link_libraries(otto ${OpenCV_LIBRARIES})

# Testing
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()
file(GLOB_RECURSE OTTO_TESTS_FILES "${CMAKE_CURRENT_LIST_DIR}/tests/*.cpp")

add_executable(otto_tests ${OTTO_TESTS_FILES})

target_link_libraries(
  otto_tests
  GTest::gtest_main
  otto
)

include(GoogleTest)
gtest_discover_tests(otto_tests)