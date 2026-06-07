cmake_minimum_required(VERSION 3.18...3.30)

project(ParticlesSimulation LANGUAGES CXX)

# ============================================================
# Build options
# ============================================================

option(PARTICLES_BUILD_OPENMP "Build the OpenMP C++ solver" ON)
option(PARTICLES_BUILD_CUDA "Build the CUDA C++ solver" ON)
option(PARTICLES_NATIVE_ARCH "Enable -march=native for host code" ON)
option(PARTICLES_FAST_MATH_CUDA "Enable --use_fast_math for CUDA code" OFF)

set(PARTICLES_CUDA_ARCHITECTURES "80" CACHE STRING
    "CUDA architectures to build for, e.g. 70;75;80;86;89;90"
)

# ============================================================
# C++ standard
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Dependencies
# ============================================================

find_package(HDF5 REQUIRED COMPONENTS C CXX)

if(PARTICLES_BUILD_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if(PARTICLES_BUILD_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)

        set(CMAKE_CUDA_ARCHITECTURES ${PARTICLES_CUDA_ARCHITECTURES})
    else()
        message(WARNING "CUDA compiler not found. Disabling CUDA target.")
        set(PARTICLES_BUILD_CUDA OFF)
    endif()
endif()

# ============================================================
# Common helper function
# ============================================================

function(particles_apply_common_settings target_name)
    target_include_directories(${target_name}
        PRIVATE
            ${HDF5_INCLUDE_DIRS}
    )

    target_compile_definitions(${target_name}
        PRIVATE
            ${HDF5_DEFINITIONS}
    )

    target_link_libraries(${target_name}
        PRIVATE
            ${HDF5_LIBRARIES}
    )

    if(MSVC)
        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/O2 /W4>
        )
    else()
        target_compile_options(${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra -Wpedantic>
        )

        if(PARTICLES_NATIVE_ARCH)
            target_compile_options(${target_name}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
            )
        endif()
    endif()
endfunction()

# ============================================================
# OpenMP C++ executable
# ============================================================

if(PARTICLES_BUILD_OPENMP)
    add_executable(particles_omp
        src/cpp/particles.cpp
    )

    particles_apply_common_settings(particles_omp)

    target_link_libraries(particles_omp
        PRIVATE
            OpenMP::OpenMP_CXX
    )
endif()

# ============================================================
# CUDA C++ executable
# ============================================================

if(PARTICLES_BUILD_CUDA)
    add_executable(particles_cuda
        src/cpp/particles.cu
    )

    particles_apply_common_settings(particles_cuda)

    set_target_properties(particles_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    if(MSVC)
        target_compile_options(particles_cuda
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )
    else()
        target_compile_options(particles_cuda
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        )

        if(PARTICLES_FAST_MATH_CUDA)
            target_compile_options(particles_cuda
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
            )
        endif()
    endif()
endif()

# ============================================================
# Convenience messages
# ============================================================

message(STATUS "CoolingSimulation configuration")
message(STATUS "  Build OpenMP solver:       ${PARTICLES_BUILD_OPENMP}")
message(STATUS "  Build CUDA solver:         ${PARTICLES_BUILD_CUDA}")
message(STATUS "  Native host architecture:  ${PARTICLES_NATIVE_ARCH}")

if(PARTICLES_BUILD_CUDA)
    message(STATUS "  CUDA architectures:        ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  CUDA fast math:            ${PARTICLES_FAST_MATH_CUDA}")
endif()

# ============================================================
# Optional install rules
# ============================================================

if(TARGET particles_omp)
    install(
        TARGETS particles_omp
        RUNTIME DESTINATION bin
    )
endif()

if(TARGET particles_cuda)
    install(
        TARGETS particles_cuda
        RUNTIME DESTINATION bin
    )
endif()

