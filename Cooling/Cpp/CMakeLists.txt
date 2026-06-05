cmake_minimum_required(VERSION 3.16...3.30)
project(CoolingSimulation LANGUAGES CXX)

# ------------------------------------------------------------
# C++ STANDARD CONFIGURATION
# ------------------------------------------------------------
# Required for std::clamp (C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------
# DEPENDENCIES
# ------------------------------------------------------------
# Find OpenMP
find_package(OpenMP REQUIRED)

# Find HDF5 with both C and C++ bindings
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# ------------------------------------------------------------
# EXECUTABLE & SOURCES
# ------------------------------------------------------------
# Assuming your code is saved as cooling.cpp
add_executable(cooling_sim cooling.cpp)

# ------------------------------------------------------------
# TARGET PROPERTIES & DEPENDENCIES
# ------------------------------------------------------------
target_link_libraries(cooling_sim 
    PRIVATE 
        OpenMP::OpenMP_CXX
        ${HDF5_LIBRARIES}
)

target_include_directories(cooling_sim 
    PRIVATE 
        ${HDF5_INCLUDE_DIRS}
)

target_compile_definitions(cooling_sim
    PRIVATE
        # Passes HDF5 definitions if any are required by the platform
        ${HDF5_DEFINITIONS}
)

# ------------------------------------------------------------
# PERFORMANCE COMPILER FLAGS
# ------------------------------------------------------------
if(MSVC)
    target_compile_options(cooling_sim PRIVATE /O2 /Oi /Ot)
else()
    # GNU / Clang high-performance flags
    target_compile_options(cooling_sim PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
endif()

