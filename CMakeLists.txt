cmake_minimum_required(VERSION 3.0.2)
project(vo)

## Compile as C++17
# add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

# add_compile_options(-ffast-math)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -ftree-vectorize")

# find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)

# Find YAML
find_package(yaml-cpp REQUIRED)

# Find Sophus
find_package(Sophus REQUIRED)
include_directories(ThirdParty/Sophus)

# Find OpenCV
find_package(OpenCV 4 REQUIRED)

# Find OpenMP
if (USE_OMP)
  find_package(OpenMP REQUIRED)
  message(STATUS "Found OpenMP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(USE_OMP)

# Enable CUDA
if (USE_GPU)
  message(STATUS "Compiling with GPU support (CUDA)")
  find_package(CUDA REQUIRED)
  enable_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "-arch=sm_53")
    set(CUDA_VERBOSE_BUILD on )
  endif(CMAKE_CUDA_COMPILER)

endif(USE_GPU)

# VO Library
file(GLOB_RECURSE VO_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/src/visual-odometry/*.cpp")

if(USE_GPU)
  file(GLOB_RECURSE VO_GPU_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/src/visual-odometry/*.cu")
  set(VO_SRC_FILES ${VO_SRC_FILES} ${VO_GPU_SRC_FILES})
endif(USE_GPU)

set(VO_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/include/visual-odometry)

add_library(vo SHARED
  ${VO_SRC_FILES}
)

target_include_directories( vo PUBLIC 
  ${VO_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${manif_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

if(USE_GPU)
  target_compile_definitions(vo PRIVATE VO_CUDA_ENABLED)
  set_target_properties(vo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(vo PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif(USE_GPU)

if (USE_OMP)
  target_compile_definitions(vo PRIVATE VO_OMP_ENABLED)
endif(USE_OMP)

target_link_libraries(
  vo
  PUBLIC
    ${OpenCV_LIBRARIES}
    Sophus::Sophus
    ${YAML_CPP_LIBRARIES}
    # $<$<BOOL:${OpenMP_FOUND}>:OpenMP::OpenMP_CXX>
)

# Python bindings
if(BUILD_PYTHON)
  message("Building Python Bindings")
  add_subdirectory(extern/pybind11)

  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "NUMPY_INCLUDE: " ${NUMPY_INCLUDE})
  include_directories(${NUMPY_INCLUDE})

  file(GLOB_RECURSE PYVO_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/python-bindings/*.cpp")
  pybind11_add_module(pyvo MODULE ${PYVO_SRC_FILES})

  target_link_libraries(pyvo PRIVATE
    vo
    ${OpenCV_LIBRARIES}
  )

  if(USE_GPU)
    target_compile_definitions(pyvo PRIVATE PYVO_CUDA_ENABLED)
  endif(USE_GPU)

  target_link_libraries(pyvo PRIVATE
    vo
    ${OpenCV_LIBRARIES}
  )

endif(BUILD_PYTHON)

