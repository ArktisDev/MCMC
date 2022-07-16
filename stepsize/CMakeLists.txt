#----------------------------------------------------------------------------
# Setup the project
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(MCMC_Stepsize_Analysis)


#----------------------------------------------------------------------------
# Set the appropriate compile options
#
add_compile_options(-Wall -O3 -Wno-unused-variable -Wno-sign-compare)

#----------------------------------------------------------------------------
# Locate sources and headers for this project
# NB: headers are included so they will show up in IDEs
#
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hpp)

#----------------------------------------------------------------------------
# Add the executables, each source is an exe as if they are scripts

foreach ( sourceFile ${sources} )
    get_filename_component(filename ${sourceFile} NAME)
    string( REPLACE ".cpp" "" source ${filename} )
    add_executable(${source} ${sourceFile} ${headers})
    # Output all binaries to the build dir
    set_target_properties(${source}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "."
    )
endforeach ( sourceFile ${sources} )

install(TARGETS)