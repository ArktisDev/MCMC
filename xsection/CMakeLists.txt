#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(MCMC)

enable_language(CUDA)

#----------------------------------------------------------------------------
# Set the appropriate compile options
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
SET(CMAKE_CXX_FLAGS -Wall -O3)
SET(CMAKE_CUDA_FLAGS "-O3 --expt-relaxed-constexpr")

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB cxxsources ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB cxxheaders ${PROJECT_SOURCE_DIR}/include/*.hpp)
file(GLOB cusources ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB cuheaders ${PROJECT_SOURCE_DIR}/include/*.cuh)

# Add executable and make it be put in build dir
add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cu ${cxxsources} ${cxxheaders} ${cusources} ${cuheaders})

# If you want to compile for specific architectures change the below line
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES OFF)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY ".")
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

install(TARGETS)