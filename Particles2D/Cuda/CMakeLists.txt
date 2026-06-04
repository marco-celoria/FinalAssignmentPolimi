cmake_minimum_required(VERSION 3.18...3.30)

project(ParticleSimulation LANGUAGES CXX CUDA)

# ------------------------------------------------------------
# C++ / CUDA STANDARD CONFIGURATION
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# GPU ARCHITECTURES
# ------------------------------------------------------------
# Set this to your target GPU architecture(s).
# Examples:
#   70 = V100
#   75 = T4
#   80 = A100
#   86 = RTX 30xx / A40
#   89 = RTX 40xx
#   90 = H100
#
# If you know your GPU, set only the needed arch for best build time.
set(CMAKE_CUDA_ARCHITECTURES 80)

# ------------------------------------------------------------
# DEPENDENCIES
# ------------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# ------------------------------------------------------------
# EXECUTABLE & SOURCES
# ------------------------------------------------------------
add_executable(particle_sim
    particles.cu
)

# ------------------------------------------------------------
# TARGET PROPERTIES
# ------------------------------------------------------------
set_target_properties(particle_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ------------------------------------------------------------
# INCLUDE DIRECTORIES
# ------------------------------------------------------------
target_include_directories(particle_sim
    PRIVATE
        ${HDF5_INCLUDE_DIRS}
)

# ------------------------------------------------------------
# COMPILE DEFINITIONS
# ------------------------------------------------------------
target_compile_definitions(particle_sim
    PRIVATE
        ${HDF5_DEFINITIONS}
)

# ------------------------------------------------------------
# LINK LIBRARIES
# ------------------------------------------------------------
target_link_libraries(particle_sim
    PRIVATE
        OpenMP::OpenMP_CXX
        ${HDF5_LIBRARIES}
)

# ------------------------------------------------------------
# COMPILER OPTIONS
# ------------------------------------------------------------
if(MSVC)
    target_compile_options(particle_sim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    )
else()
    target_compile_options(particle_sim PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
endif()
