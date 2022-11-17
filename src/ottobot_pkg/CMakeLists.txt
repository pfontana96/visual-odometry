cmake_minimum_required(VERSION 3.0.2)
project(ottobot_pkg)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS 
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_filters
  tf2
  tf2_ros
  tf2_eigen
  pcl_ros
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)

# Find PCL library
find_package(PCL REQUIRED COMPONENTS common io)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Find yaml-cpp
find_package(yaml-cpp REQUIRED)

# Import OpenCV library
find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
  # include_directories(${OpenCV_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "OpenCV > 4 not found.")
endif(OpenCV_FOUND)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
  message(STATUS "Found OpenMP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OpenMP_FOUND)

# Enable CUDA
enable_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  message(STATUS "Found CUDA ${CMAKE_CUDA_COMPILER_VERSION}")
  set(CMAKE_CUDA_FLAGS "-arch=sm_53")
  set(CUDA_VERBOSE_BUILD on )
endif(CMAKE_CUDA_COMPILER)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES dvo
#  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs
  DEPENDS OpenCV
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(
#   # include
#   ${catkin_INCLUDE_DIRS}
# )

## Declare a C++ library

# DVO Library
file(GLOB DVO_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/src/dvo/*.cpp")

if(CMAKE_CUDA_COMPILER)
  file(GLOB dvo_gpu_src_files "${CMAKE_CURRENT_LIST_DIR}/src/dvo/*.cu")
  set(DVO_SRC_FILES ${DVO_SRC_FILES} ${dvo_gpu_src_files})
endif(CMAKE_CUDA_COMPILER)
# message(STATUS "DVO src: ${DVO_SRC_FILES}")

set(DVO_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/dvo)
add_library(dvo ${DVO_SRC_FILES})
target_include_directories( dvo PUBLIC 
  ${DVO_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${manif_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)
if(CMAKE_CUDA_COMPILER)
  set_target_properties(dvo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(dvo PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif(CMAKE_CUDA_COMPILER)
target_link_libraries(dvo ${OpenCV_LIBRARIES} yaml-cpp)

if(CATKIN_ENABLE_TESTING)
  message(STATUS "dvo tests enabled")
  catkin_add_gtest(test_dvo test/dvo_tests.cpp)
  # target_include_directories(test_dvo PRIVATE 
  #   ${DVO_INCLUDE_DIRS})
  target_link_libraries(test_dvo dvo yaml-cpp)
endif()

## Nodes

# dvo_node
add_executable(dvo_node src/node_dvo.cpp)
target_include_directories(dvo_node PRIVATE
  ${catkin_INCLUDE_DIRS}
  ${DVO_INCLUDE_DIR}
  ${PCL_INCUDE_DIRS}
  include
)

target_compile_options(dvo_node PRIVATE ${OpenMP_FLAGS})

target_link_libraries(dvo_node 
  ${catkin_LIBRARIES} 
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES} 
  cudart
  dvo
)
add_dependencies(dvo_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} dvo)

