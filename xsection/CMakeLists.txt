#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(MCMC)

enable_language(CUDA)

#----------------------------------------------------------------------------
# Set the appropriate compile options
#
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
SET(CMAKE_CXX_FLAGS -Wall -O3)
SET(CMAKE_CUDA_FLAGS "-O3 --expt-relaxed-constexpr")

#
# To be able to specify a certain macro file to include in the 
# hacky way I am including them
#

# Usage: -DUSE_MACRO_FILE="ON"
IF(USE_MACRO_FILE STREQUAL "ON")
    ADD_DEFINITIONS(-DUSE_MACRO_FILE)
ENDIF()

# Usage: -DMACRO_FILE_PATH=\"path\"
IF (NOT DEFINED MACRO_FILE_PATH)
    SET(MACRO_FILE_PATH \"/NO_INPUT_FILE\")
ENDIF()
ADD_DEFINITIONS(-DMACRO_FILE_PATH=${MACRO_FILE_PATH})

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
set_target_properties(main PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

install(TARGETS)